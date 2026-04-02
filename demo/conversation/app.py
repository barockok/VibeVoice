"""Live conversation server: mic audio in, ASR, response, TTS audio out."""

import asyncio
import datetime
import json
import threading
import traceback
from pathlib import Path
from typing import cast

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState

from demo.conversation.asr_service import ASRService
from demo.conversation.tts_service import TTSService
from demo.conversation.response_generator import TemplateResponseGenerator

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

    asr_model = os.environ.get("ASR_MODEL_PATH", "microsoft/VibeVoice-ASR-0.5B")
    tts_model = os.environ.get("TTS_MODEL_PATH", "microsoft/VibeVoice-Realtime-0.5B")
    voices_dir = os.environ.get("VOICES_DIR", str(BASE.parent / "voices" / "streaming_model"))
    device = os.environ.get("MODEL_DEVICE", "cuda")

    asr = ASRService(model_path=asr_model, device=device)
    asr.load()
    app.state.asr = asr

    tts = TTSService(model_path=tts_model, voices_dir=voices_dir, device=device)
    tts.load()
    app.state.tts = tts

    app.state.responder = TemplateResponseGenerator()
    app.state.lock = asyncio.Lock()
    print("[startup] Conversation server ready.")


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/config")
def config():
    tts: TTSService = app.state.tts
    return {"voices": tts.get_voices(), "default_voice": tts.default_voice_key}


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
