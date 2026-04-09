"""ASR service wrapping NVIDIA Parakeet TDT 0.6B model via NeMo."""

import os
import tempfile

import numpy as np
import soundfile as sf
from typing import Optional


class ParakeetASRService:
    """Wraps NVIDIA Parakeet TDT 0.6B for single-utterance transcription."""

    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.device = device
        self.model = None

    def load(self) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        # Disable cuDNN globally — some community GPU nodes have broken
        # cuDNN installations that cause CUDNN_STATUS_NOT_INITIALIZED.
        # Conv ops fall back to native CUDA kernels (slightly slower but works).
        torch.backends.cudnn.enabled = False

        print(f"[ASR] Loading Parakeet model: {self.model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_name,
            map_location="cpu",
        )
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        print("[ASR] Parakeet model loaded.")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> str:
        if self.model is None:
            raise RuntimeError("ASR service not loaded")

        import librosa

        # Parakeet expects 16 kHz mono audio
        target_sr = 16000
        if sample_rate != target_sr:
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=target_sr,
            )

        # NeMo transcribe API takes file paths, so write a temp wav file
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(tmp_fd)
            sf.write(tmp_path, audio, target_sr)

            results = self.model.transcribe([tmp_path])

            # NeMo returns a list of strings (or Hypothesis objects depending
            # on version). Handle both cases.
            if isinstance(results, list) and len(results) > 0:
                result = results[0]
                # If it's a Hypothesis object, extract the text attribute
                if hasattr(result, "text"):
                    return result.text.strip()
                if isinstance(result, str):
                    return result.strip()

            return ""
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
