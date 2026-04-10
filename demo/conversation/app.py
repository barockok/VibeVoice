"""Live conversation server: mic audio in, ASR, response, TTS audio out."""

import asyncio
import datetime
import json
import re
import threading
import traceback
from pathlib import Path
from typing import cast

import numpy as np
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from demo.conversation.response_generator import TemplateResponseGenerator

# Patterns that indicate noise/non-speech from ASR
NOISE_PATTERNS = re.compile(
    r"^\[?(Noise|Silence|Unintelligible Speech|Music|Applause|Laughter)\]?$",
    re.IGNORECASE,
)

SAMPLE_RATE = 24_000
BASE = Path(__file__).parent

app = FastAPI()


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


async def ws_send_json(ws: WebSocket, data: dict):
    try:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_text(json.dumps({**data, "timestamp": timestamp()}))
    except Exception:
        pass


@app.on_event("startup")
async def startup():
    import os

    stt_engine = os.environ.get("STT_ENGINE", "parakeet")
    llm_engine = os.environ.get("LLM_ENGINE", "qwen")
    parakeet_model = os.environ.get("PARAKEET_MODEL", "nvidia/parakeet-tdt-0.6b-v3")
    qwen_model = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    asr_model = os.environ.get("ASR_MODEL_PATH", "microsoft/VibeVoice-ASR")
    tts_model = os.environ.get("TTS_MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B")
    voices_dir = os.environ.get("VOICES_DIR", str(BASE.parent / "voices" / "streaming_model"))
    device = os.environ.get("MODEL_DEVICE", "cuda")

    # --- STT engine ---
    if stt_engine == "parakeet":
        from demo.conversation.parakeet_asr_service import ParakeetASRService

        asr = ParakeetASRService(model_name=parakeet_model, device=device)
    else:
        from demo.conversation.asr_service import ASRService
        asr = ASRService(model_path=asr_model, device=device)
    asr.load()
    app.state.asr = asr

    # --- TTS ---
    from demo.conversation.tts_service import TTSService
    tts = TTSService(model_path=tts_model, voices_dir=voices_dir, device=device)
    tts.load()
    app.state.tts = tts

    # --- LLM response generator ---
    if llm_engine == "qwen":
        from demo.conversation.qwen_response_generator import QwenResponseGenerator

        responder = QwenResponseGenerator(model_name=qwen_model, device=device)
        responder.load()
    else:
        responder = TemplateResponseGenerator()
    app.state.responder = responder

    app.state.lock = asyncio.Lock()
    print("[startup] Conversation server ready.")


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/health")
def health():
    asr_ok = hasattr(app.state, "asr") and app.state.asr.model is not None
    tts_ok = hasattr(app.state, "tts") and app.state.tts.model is not None
    llm_ok = hasattr(app.state, "responder") and app.state.responder is not None
    all_ok = asr_ok and tts_ok and llm_ok
    return {
        "status": "ok" if all_ok else "degraded",
        "models": {
            "asr": "loaded" if asr_ok else "not_loaded",
            "tts": "loaded" if tts_ok else "not_loaded",
            "vad": "loaded",  # VAD is handled by Rust server
            "llm": "loaded" if llm_ok else "not_loaded",
        },
    }


@app.get("/config")
def config():
    tts = app.state.tts
    return {"voices": tts.get_voices(), "default_voice": tts.default_voice_key}


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if NOISE_PATTERNS.match(stripped):
        return True
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    return False


# ── HTTP API (sidecar-compatible with OttrVoice Rust server) ──


@app.post("/api/stt")
async def stt(request: Request):
    """Transcribe a complete utterance (raw PCM16 bytes). Returns JSON with transcription + LLM quick reply."""
    asr = app.state.asr
    responder = app.state.responder

    body = await request.body()
    if not body:
        return JSONResponse({"text": "", "cleaned_text": "", "quick_reply": "", "is_noise": True, "timing": None})

    pcm16 = np.frombuffer(body, dtype=np.int16)
    audio_float = pcm16.astype(np.float32) / 32768.0

    # Skip very short audio
    if audio_float.size < SAMPLE_RATE * 0.3:
        return JSONResponse({"text": "", "cleaned_text": "", "quick_reply": "", "is_noise": True, "timing": None})

    try:
        transcription = await asyncio.to_thread(asr.transcribe, audio_float, SAMPLE_RATE)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"ASR error: {e}"}, status_code=500)

    if _is_noise(transcription):
        return JSONResponse({"text": "", "cleaned_text": "", "quick_reply": "", "is_noise": True, "timing": None})

    # Use Qwen LLM for cleaning + quick reply generation
    call_context = request.headers.get("x-call-context", "")
    call_tone = request.headers.get("x-call-tone", "")

    try:
        # Build a prompt for transcript cleaning + quick reply
        quick_reply = await asyncio.to_thread(
            _generate_quick_reply, responder, transcription, call_context, call_tone
        )
    except Exception as e:
        traceback.print_exc()
        quick_reply = ""

    return JSONResponse({
        "text": transcription,
        "cleaned_text": transcription,  # Parakeet output is already clean
        "quick_reply": quick_reply,
        "is_noise": False,
        "timing": None,
    })


def _generate_quick_reply(responder, transcription: str, context: str, tone: str) -> str:
    """Use Qwen to generate a brief filler/acknowledgment reply."""
    if not hasattr(responder, 'generate'):
        return ""

    prompt = f"The user said: \"{transcription}\""
    if context:
        prompt += f"\nContext: {context}"
    if tone:
        prompt += f"\nTone: {tone}"
    prompt += "\nGenerate a brief (3-8 word) natural acknowledgment or filler response."

    # Use the responder's generate method with empty history for quick reply
    try:
        reply = responder.generate(prompt, [])
        # Truncate if too long
        words = reply.split()
        if len(words) > 10:
            reply = " ".join(words[:8])
        return reply
    except Exception:
        return ""


# ── Streaming STT (session-based) ──

import uuid as _uuid

# In-memory session store: session_id -> accumulated audio (float32 ndarray)
_stt_sessions: dict = {}


@app.post("/api/stt/stream/start")
async def stt_stream_start():
    """Start a streaming STT session. Returns session_id."""
    session_id = str(_uuid.uuid4())[:8]
    _stt_sessions[session_id] = {"audio": np.array([], dtype=np.float32), "text": ""}
    return JSONResponse({"session_id": session_id})


@app.post("/api/stt/stream/chunk/{session_id}")
async def stt_stream_chunk(session_id: str, request: Request):
    """Send an audio chunk to an active STT session. Returns current partial transcript."""
    if session_id not in _stt_sessions:
        return JSONResponse({"error": "session not found"}, status_code=404)

    asr = app.state.asr
    session = _stt_sessions[session_id]

    body = await request.body()
    if not body:
        return JSONResponse({"text": session["text"]})

    pcm16 = np.frombuffer(body, dtype=np.int16)
    audio_float = pcm16.astype(np.float32) / 32768.0

    # Append to accumulated audio
    session["audio"] = np.concatenate([session["audio"], audio_float])

    # Transcribe the full accumulated audio
    if session["audio"].size < SAMPLE_RATE * 0.3:
        return JSONResponse({"text": ""})

    try:
        transcription = await asyncio.to_thread(asr.transcribe, session["audio"], SAMPLE_RATE)
        session["text"] = transcription if not _is_noise(transcription) else session["text"]
    except Exception as e:
        traceback.print_exc()
        # Keep previous text on error
        pass

    return JSONResponse({"text": session["text"]})


@app.post("/api/stt/stream/end/{session_id}")
async def stt_stream_end(session_id: str, request: Request):
    """Finalize a streaming STT session. Returns final transcript + quick reply."""
    if session_id not in _stt_sessions:
        return JSONResponse({"error": "session not found"}, status_code=404)

    session = _stt_sessions.pop(session_id)
    text = session["text"]

    if not text.strip():
        return JSONResponse({"text": "", "quick_reply": "", "is_noise": True})

    # Generate quick reply using Qwen
    call_context = request.headers.get("x-call-context", "")
    call_tone = request.headers.get("x-call-tone", "")

    responder = app.state.responder
    try:
        quick_reply = await asyncio.to_thread(
            _generate_quick_reply, responder, text, call_context, call_tone
        )
    except Exception:
        quick_reply = ""

    return JSONResponse({
        "text": text,
        "quick_reply": quick_reply,
        "is_noise": False,
    })


@app.post("/api/tts")
async def tts_endpoint(request: Request):
    """Synthesize text to speech, streaming PCM16 audio bytes."""
    tts = app.state.tts

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    text = data.get("text", "")
    voice = data.get("voice", None)

    if not text.strip():
        return StreamingResponse(iter([]), media_type="application/octet-stream")

    stop_event = threading.Event()

    async def generate():
        try:
            iterator = tts.stream(text, voice_key=voice, stop_event=stop_event)
            sentinel = object()
            while True:
                if await request.is_disconnected():
                    stop_event.set()
                    break
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                payload = tts.chunk_to_pcm16(cast(np.ndarray, chunk))
                yield payload
        except Exception:
            traceback.print_exc()
            stop_event.set()
        finally:
            stop_event.set()

    return StreamingResponse(generate(), media_type="application/octet-stream")


@app.websocket("/ws")
async def conversation_ws(ws: WebSocket):
    await ws.accept()
    asr: ASRService = app.state.asr
    tts: TTSService = app.state.tts
    responder = app.state.responder
    lock: asyncio.Lock = app.state.lock

    audio_buffer = bytearray()
    voice_key = None
    history = []

    await ws_send_json(ws, {"type": "status", "state": "listening"})

    try:
        while ws.client_state == WebSocketState.CONNECTED:
            message = await ws.receive()

            if "bytes" in message and message["bytes"]:
                audio_buffer.extend(message["bytes"])
                continue

            if "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "config":
                    voice_key = data.get("voice")
                    continue

                if msg_type == "start_recording":
                    audio_buffer.clear()
                    continue

                if msg_type == "stop_recording":
                    if not audio_buffer:
                        await ws_send_json(ws, {"type": "error", "message": "No audio received"})
                        await ws_send_json(ws, {"type": "status", "state": "listening"})
                        continue

                    if lock.locked():
                        await ws_send_json(ws, {"type": "error", "message": "Still processing previous request"})
                        continue

                    async with lock:
                        await ws_send_json(ws, {"type": "status", "state": "processing"})

                        pcm16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                        audio_float = pcm16.astype(np.float32) / 32768.0
                        audio_buffer.clear()

                        if audio_float.size < SAMPLE_RATE * 0.3:
                            await ws_send_json(ws, {"type": "error", "message": "Audio too short (< 0.3s)"})
                            await ws_send_json(ws, {"type": "status", "state": "listening"})
                            continue

                        try:
                            transcription = await asyncio.to_thread(
                                asr.transcribe, audio_float, SAMPLE_RATE
                            )
                        except Exception as e:
                            traceback.print_exc()
                            await ws_send_json(ws, {"type": "error", "message": f"ASR error: {e}"})
                            await ws_send_json(ws, {"type": "status", "state": "listening"})
                            continue

                        await ws_send_json(ws, {"type": "transcription", "text": transcription})

                        if not transcription.strip():
                            await ws_send_json(ws, {"type": "status", "state": "listening"})
                            continue

                        history.append({"role": "user", "text": transcription})
                        response_text = responder.generate(transcription, history)
                        history.append({"role": "assistant", "text": response_text})
                        await ws_send_json(ws, {"type": "response", "text": response_text})

                        await ws_send_json(ws, {"type": "status", "state": "speaking"})
                        await ws_send_json(ws, {"type": "tts_start"})

                        stop_event = threading.Event()
                        try:
                            iterator = tts.stream(
                                response_text, voice_key=voice_key, stop_event=stop_event
                            )
                            sentinel = object()
                            while ws.client_state == WebSocketState.CONNECTED:
                                chunk = await asyncio.to_thread(next, iterator, sentinel)
                                if chunk is sentinel:
                                    break
                                payload = tts.chunk_to_pcm16(cast(np.ndarray, chunk))
                                await ws.send_bytes(payload)
                        except WebSocketDisconnect:
                            stop_event.set()
                            raise
                        except Exception as e:
                            traceback.print_exc()
                            stop_event.set()
                            await ws_send_json(ws, {"type": "error", "message": f"TTS error: {e}"})
                        finally:
                            stop_event.set()

                        await ws_send_json(ws, {"type": "tts_end"})
                        await ws_send_json(ws, {"type": "status", "state": "listening"})

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
    finally:
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
        except Exception:
            pass
