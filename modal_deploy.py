"""
Modal deployment for VibeVoice Realtime TTS Demo.

Usage:
    modal deploy modal_deploy.py
    modal serve modal_deploy.py   # for development
"""

import modal

app = modal.App("vibevoice-realtime")


def download_model():
    from huggingface_hub import snapshot_download
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
        download_model,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .add_local_dir("demo/voices/streaming_model", remote_path="/voices/streaming_model")
    .add_local_dir("demo/web", remote_path="/web")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=5)
@modal.asgi_app()
def serve():
    import datetime
    import asyncio
    import json
    import os
    import threading
    import traceback
    import copy
    from pathlib import Path
    from queue import Empty, Queue
    from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

    import numpy as np
    import torch
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import FileResponse, HTMLResponse
    from starlette.websockets import WebSocketDisconnect, WebSocketState

    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
    from vibevoice.modular.streamer import AudioStreamer

    SAMPLE_RATE = 24_000
    MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
    VOICES_DIR = Path("/voices/streaming_model")
    WEB_DIR = Path("/web")

    def get_timestamp():
        return datetime.datetime.utcnow().replace(
            tzinfo=datetime.timezone.utc
        ).astimezone(
            datetime.timezone(datetime.timedelta(hours=8))
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    class StreamingTTSService:
        def __init__(self, model_path, device="cuda", inference_steps=5):
            self.model_path = model_path
            self.inference_steps = inference_steps
            self.sample_rate = SAMPLE_RATE
            self.processor = None
            self.model = None
            self.voice_presets = {}
            self.default_voice_key = None
            self._voice_cache = {}
            self.device = device
            self._torch_device = torch.device(device)

        def load(self):
            print(f"[startup] Loading processor from {self.model_path}")
            self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
            print(f"Using device: cuda, dtype: {load_dtype}, attn: {attn_impl}")

            try:
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                )
            except Exception:
                print("flash_attention_2 failed, falling back to sdpa")
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map="cuda",
                    attn_implementation="sdpa",
                )

            self.model.eval()
            self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
                self.model.model.noise_scheduler.config,
                algorithm_type="sde-dpmsolver++",
                beta_schedule="squaredcos_cap_v2",
            )
            self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

            self.voice_presets = self._load_voice_presets()
            self.default_voice_key = self._determine_voice_key(None)
            self._ensure_voice_cached(self.default_voice_key)

        def _load_voice_presets(self):
            presets = {}
            for pt_path in VOICES_DIR.rglob("*.pt"):
                presets[pt_path.stem] = pt_path
            if not presets:
                raise RuntimeError(f"No voice presets found in {VOICES_DIR}")
            print(f"[startup] Found {len(presets)} voice presets")
            return dict(sorted(presets.items()))

        def _determine_voice_key(self, name):
            if name and name in self.voice_presets:
                return name
            default_key = "en-Carter_man"
            if default_key in self.voice_presets:
                return default_key
            return next(iter(self.voice_presets))

        def _ensure_voice_cached(self, key):
            if key not in self.voice_presets:
                raise RuntimeError(f"Voice preset {key!r} not found")
            if key not in self._voice_cache:
                preset_path = self.voice_presets[key]
                print(f"[startup] Loading voice preset {key}")
                self._voice_cache[key] = torch.load(
                    preset_path, map_location=self._torch_device, weights_only=False,
                )
            return self._voice_cache[key]

        def _get_voice_resources(self, requested_key):
            key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
            if key is None:
                key = next(iter(self.voice_presets))
                self.default_voice_key = key
            prefilled = self._ensure_voice_cached(key)
            return key, prefilled

        def _prepare_inputs(self, text, prefilled_outputs):
            processed = self.processor.process_input_with_cached_prompt(
                text=text.strip(),
                cached_prompt=prefilled_outputs,
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            return {
                k: v.to(self._torch_device) if hasattr(v, "to") else v
                for k, v in processed.items()
            }

        def _run_generation(self, inputs, audio_streamer, errors, cfg_scale,
                           do_sample, temperature, top_p, refresh_negative,
                           prefilled_outputs, stop_event):
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={
                        "do_sample": do_sample,
                        "temperature": temperature if do_sample else 1.0,
                        "top_p": top_p if do_sample else 1.0,
                    },
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop_event.is_set,
                    verbose=False,
                    refresh_negative=refresh_negative,
                    all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
                )
            except Exception as exc:
                errors.append(exc)
                traceback.print_exc()
                audio_streamer.end()

        def stream(self, text, cfg_scale=1.5, do_sample=False, temperature=0.9,
                   top_p=0.9, refresh_negative=True, inference_steps=None,
                   voice_key=None, log_callback=None, stop_event=None):
            if not text.strip():
                return
            text = text.replace("\u2019", "'")
            selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

            def emit(event, **payload):
                if log_callback:
                    try:
                        log_callback(event, **payload)
                    except Exception:
                        pass

            steps_to_use = self.inference_steps
            if inference_steps is not None:
                try:
                    parsed = int(inference_steps)
                    if parsed > 0:
                        steps_to_use = parsed
                except (TypeError, ValueError):
                    pass
            if self.model:
                self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
            self.inference_steps = steps_to_use

            inputs = self._prepare_inputs(text, prefilled_outputs)
            audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            errors = []
            stop_signal = stop_event or threading.Event()

            thread = threading.Thread(
                target=self._run_generation,
                kwargs=dict(
                    inputs=inputs, audio_streamer=audio_streamer, errors=errors,
                    cfg_scale=cfg_scale, do_sample=do_sample, temperature=temperature,
                    top_p=top_p, refresh_negative=refresh_negative,
                    prefilled_outputs=prefilled_outputs, stop_event=stop_signal,
                ),
                daemon=True,
            )
            thread.start()

            generated_samples = 0
            try:
                stream = audio_streamer.get_stream(0)
                for audio_chunk in stream:
                    if torch.is_tensor(audio_chunk):
                        audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                    else:
                        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
                    if audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk.reshape(-1)
                    peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                    if peak > 1.0:
                        audio_chunk = audio_chunk / peak
                    generated_samples += int(audio_chunk.size)
                    emit("model_progress",
                         generated_sec=generated_samples / self.sample_rate,
                         chunk_sec=audio_chunk.size / self.sample_rate)
                    yield audio_chunk.astype(np.float32, copy=False)
            finally:
                stop_signal.set()
                audio_streamer.end()
                thread.join()
                if errors:
                    emit("generation_error", message=str(errors[0]))
                    raise errors[0]

        def chunk_to_pcm16(self, chunk):
            chunk = np.clip(chunk, -1.0, 1.0)
            return (chunk * 32767.0).astype(np.int16).tobytes()

    # --- Build FastAPI app ---
    web_app = FastAPI()

    @web_app.on_event("startup")
    async def _startup():
        service = StreamingTTSService(model_path=MODEL_PATH, device="cuda")
        service.load()
        web_app.state.tts_service = service
        web_app.state.websocket_lock = asyncio.Lock()
        print("[startup] Model ready.")

    @web_app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @web_app.get("/config")
    def get_config():
        service = web_app.state.tts_service
        voices = sorted(service.voice_presets.keys())
        return {"voices": voices, "default_voice": service.default_voice_key}

    @web_app.websocket("/stream")
    async def websocket_stream(ws: WebSocket):
        await ws.accept()
        text = ws.query_params.get("text", "")
        print(f"Client connected, text={text!r}")
        cfg_param = ws.query_params.get("cfg")
        steps_param = ws.query_params.get("steps")
        voice_param = ws.query_params.get("voice")

        try:
            cfg_scale = float(cfg_param) if cfg_param is not None else 1.5
        except ValueError:
            cfg_scale = 1.5
        if cfg_scale <= 0:
            cfg_scale = 1.5
        try:
            inference_steps = int(steps_param) if steps_param is not None else None
            if inference_steps is not None and inference_steps <= 0:
                inference_steps = None
        except ValueError:
            inference_steps = None

        service = web_app.state.tts_service
        lock = web_app.state.websocket_lock

        if lock.locked():
            busy_msg = {
                "type": "log", "event": "backend_busy",
                "data": {"message": "Please wait for the other requests to complete."},
                "timestamp": get_timestamp(),
            }
            try:
                await ws.send_text(json.dumps(busy_msg))
            except Exception:
                pass
            await ws.close(code=1013, reason="Service busy")
            return

        acquired = False
        try:
            await lock.acquire()
            acquired = True

            log_queue = Queue()

            def enqueue_log(event, **data):
                log_queue.put({"event": event, "data": data})

            async def flush_logs():
                while True:
                    try:
                        entry = log_queue.get_nowait()
                    except Empty:
                        break
                    msg = {
                        "type": "log", "event": entry.get("event"),
                        "data": entry.get("data", {}),
                        "timestamp": get_timestamp(),
                    }
                    try:
                        await ws.send_text(json.dumps(msg))
                    except Exception:
                        break

            enqueue_log("backend_request_received",
                        text_length=len(text or ""), cfg_scale=cfg_scale,
                        inference_steps=inference_steps, voice=voice_param)

            stop_signal = threading.Event()
            iterator = service.stream(
                text, cfg_scale=cfg_scale, inference_steps=inference_steps,
                voice_key=voice_param, log_callback=enqueue_log,
                stop_event=stop_signal,
            )
            sentinel = object()
            first_ws_send_logged = False

            await flush_logs()

            try:
                while ws.client_state == WebSocketState.CONNECTED:
                    await flush_logs()
                    chunk = await asyncio.to_thread(next, iterator, sentinel)
                    if chunk is sentinel:
                        break
                    chunk = cast(np.ndarray, chunk)
                    payload = service.chunk_to_pcm16(chunk)
                    await ws.send_bytes(payload)
                    if not first_ws_send_logged:
                        first_ws_send_logged = True
                        enqueue_log("backend_first_chunk_sent")
                    await flush_logs()
            except WebSocketDisconnect:
                print("Client disconnected")
                enqueue_log("client_disconnected")
                stop_signal.set()
            except Exception as e:
                print(f"Error in websocket stream: {e}")
                traceback.print_exc()
                enqueue_log("backend_error", message=str(e))
                stop_signal.set()
            finally:
                stop_signal.set()
                enqueue_log("backend_stream_complete")
                await flush_logs()
                try:
                    close_fn = getattr(iterator, "close", None)
                    if callable(close_fn):
                        close_fn()
                except Exception:
                    pass
                while not log_queue.empty():
                    try:
                        log_queue.get_nowait()
                    except Empty:
                        break
                try:
                    if ws.client_state == WebSocketState.CONNECTED:
                        await ws.close()
                except Exception as e:
                    print(f"Error closing websocket: {e}")
                print("WS handler exit")
        finally:
            if acquired:
                lock.release()

    return web_app
