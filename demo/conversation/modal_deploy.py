"""
Modal deployment for VibeVoice Live Conversation Demo.

Usage:
    modal deploy demo/conversation/modal_deploy.py
    modal serve demo/conversation/modal_deploy.py   # for development
"""

import modal

app = modal.App("vibevoice-conversation")


def download_models():
    from huggingface_hub import snapshot_download
    snapshot_download("microsoft/VibeVoice-ASR")
    snapshot_download("microsoft/VibeVoice-Realtime-0.5B")


# Build the container image with all dependencies
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.51.3,<5.0.0",
        "accelerate",
        "diffusers",
        "numpy",
        "scipy",
        "librosa",
        "ml-collections",
        "absl-py",
        "gradio",
        "av",
        "aiortc",
        "uvicorn[standard]",
        "fastapi",
        "pydub",
        "requests",
        "soundfile",
        "packaging",
        "ninja",
        "wheel",
        "setuptools",
    )
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
    )
    .pip_install("vibevoice@git+https://github.com/microsoft/VibeVoice.git")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("hf_transfer")
    .run_function(
        download_models,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("demo/voices/streaming_model", remote_path="/voices/streaming_model")
    .add_local_file("demo/conversation/asr_service.py", remote_path="/app/asr_service.py")
    .add_local_file("demo/conversation/tts_service.py", remote_path="/app/tts_service.py")
    .add_local_file("demo/conversation/response_generator.py", remote_path="/app/response_generator.py")
    .add_local_file("demo/conversation/index.html", remote_path="/app/index.html")
)


@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=5)
@modal.asgi_app()
def serve():
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

    import sys
    sys.path.insert(0, "/app")

    from asr_service import ASRService
    from tts_service import TTSService
    from response_generator import TemplateResponseGenerator

    SAMPLE_RATE = 24_000
    ASR_MODEL = "microsoft/VibeVoice-ASR"
    TTS_MODEL = "microsoft/VibeVoice-Realtime-0.5B"
    VOICES_DIR = "/voices/streaming_model"
    INDEX_HTML = Path("/app/index.html")

    def timestamp():
        return datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

    async def ws_send_json(ws: WebSocket, data: dict):
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(json.dumps({**data, "timestamp": timestamp()}))
        except Exception:
            pass

    # --- Build FastAPI app ---
    web_app = FastAPI()

    @web_app.on_event("startup")
    async def _startup():
        asr = ASRService(model_path=ASR_MODEL, device="cuda")
        asr.load()
        web_app.state.asr = asr

        tts = TTSService(model_path=TTS_MODEL, voices_dir=VOICES_DIR, device="cuda")
        tts.load()
        web_app.state.tts = tts

        web_app.state.responder = TemplateResponseGenerator()
        web_app.state.lock = asyncio.Lock()
        print("[startup] Conversation server ready.")

    @web_app.get("/")
    def index():
        return FileResponse(INDEX_HTML)

    @web_app.get("/config")
    def config():
        tts: TTSService = web_app.state.tts
        return {"voices": tts.get_voices(), "default_voice": tts.default_voice_key}

    @web_app.websocket("/ws")
    async def conversation_ws(ws: WebSocket):
        await ws.accept()
        asr: ASRService = web_app.state.asr
        tts: TTSService = web_app.state.tts
        responder = web_app.state.responder
        lock: asyncio.Lock = web_app.state.lock

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

    return web_app
