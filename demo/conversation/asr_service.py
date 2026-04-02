"""ASR service wrapping VibeVoice ASR model."""

import numpy as np
import torch
from typing import Optional

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


class ASRService:
    """Wraps VibeVoice ASR for single-utterance transcription."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.processor: Optional[VibeVoiceASRProcessor] = None
        self.model: Optional[VibeVoiceASRForConditionalGeneration] = None

    def load(self) -> None:
        print(f"[ASR] Loading processor from {self.model_path}")
        self.processor = VibeVoiceASRProcessor.from_pretrained(
            self.model_path,
            language_model_pretrained_name="Qwen/Qwen2.5-7B",
        )

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn = "flash_attention_2" if self.device == "cuda" else "sdpa"

        try:
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=self.device,
                attn_implementation=attn,
                trust_remote_code=True,
            )
        except Exception:
            print("[ASR] flash_attention_2 failed, falling back to sdpa")
            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=self.device,
                attn_implementation="sdpa",
                trust_remote_code=True,
            )

        self.model.eval()
        print("[ASR] Model loaded.")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
        max_new_tokens: int = 512,
    ) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("ASR service not loaded")

        inputs = self.processor(
            audio=[audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        )
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.0,
            )

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        raw = self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # ASR model outputs structured JSON like:
        # [{"Start":0,"End":5.38,"Speaker":0,"Content":"Hello world"}]
        # Extract just the text content.
        try:
            import json
            segments = json.loads(raw)
            if isinstance(segments, list):
                return " ".join(seg.get("Content", "") for seg in segments).strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return raw
