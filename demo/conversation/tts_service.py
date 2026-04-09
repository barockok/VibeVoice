"""TTS service wrapping VibeVoice Realtime streaming model."""

import copy
import threading
import traceback
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import torch

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

SAMPLE_RATE = 24_000


class TTSService:
    """Wraps VibeVoice Realtime TTS for streaming audio generation."""

    def __init__(
        self,
        model_path: str,
        voices_dir: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ):
        self.model_path = model_path
        self.voices_dir = Path(voices_dir)
        self.device = device
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE
        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, object] = {}
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[TTS] Loading from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn = "flash_attention_2" if self.device == "cuda" else "sdpa"

        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path, torch_dtype=dtype, device_map=self.device,
                attn_implementation=attn,
            )
        except Exception:
            print("[TTS] flash_attention_2 failed, falling back to sdpa")
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path, torch_dtype=dtype, device_map=self.device,
                attn_implementation="sdpa",
            )

        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = {}
        for pt in self.voices_dir.rglob("*.pt"):
            self.voice_presets[pt.stem] = pt
        if not self.voice_presets:
            raise RuntimeError(f"No voice presets in {self.voices_dir}")
        self.voice_presets = dict(sorted(self.voice_presets.items()))

        self.default_voice_key = "en-Carter_man" if "en-Carter_man" in self.voice_presets else next(iter(self.voice_presets))
        self._ensure_cached(self.default_voice_key)
        print(f"[TTS] Loaded {len(self.voice_presets)} voices, default={self.default_voice_key}")

    def _ensure_cached(self, key: str) -> object:
        if key not in self._voice_cache:
            path = self.voice_presets[key]
            self._voice_cache[key] = torch.load(path, map_location=self._torch_device, weights_only=False)
        return self._voice_cache[key]

    def stream(
        self,
        text: str,
        voice_key: Optional[str] = None,
        cfg_scale: float = 1.5,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return

        key = voice_key if voice_key and voice_key in self.voice_presets else self.default_voice_key
        prefilled = self._ensure_cached(key)

        processed = self.processor.process_input_with_cached_prompt(
            text=text.strip().replace("\u2019", "'"),
            cached_prompt=prefilled,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            k: v.to(self._torch_device) if hasattr(v, "to") else v
            for k, v in processed.items()
        }

        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors = []
        stop = stop_event or threading.Event()

        def run():
            try:
                self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False, "temperature": 1.0, "top_p": 1.0},
                    audio_streamer=audio_streamer,
                    stop_check_fn=stop.is_set,
                    verbose=False,
                    refresh_negative=True,
                    all_prefilled_outputs=copy.deepcopy(prefilled),
                )
            except Exception as e:
                errors.append(e)
                traceback.print_exc()
                audio_streamer.end()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        try:
            for chunk in audio_streamer.get_stream(0):
                if torch.is_tensor(chunk):
                    chunk = chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    chunk = np.asarray(chunk, dtype=np.float32)
                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)
                peak = np.max(np.abs(chunk)) if chunk.size else 0.0
                if peak > 1.0:
                    chunk = chunk / peak
                yield chunk
        finally:
            stop.set()
            audio_streamer.end()
            thread.join()
            if errors:
                raise errors[0]

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        return (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def get_voices(self):
        return sorted(self.voice_presets.keys())
