# RunPod Deployment: Parakeet STT + Qwen LLM + VibeVoice TTS

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy a real-time voice conversation demo on RunPod RTX 3090 (~$0.22/hr) using Parakeet TDT 0.6B for STT, Qwen2.5-7B-Instruct for chat LLM, and VibeVoice TTS 0.5B for speech synthesis.

**Architecture:** Three models share one GPU (24GB RTX 3090). Browser sends audio via WebSocket -> Parakeet transcribes (16kHz) -> Qwen generates response -> VibeVoice TTS streams audio back. Docker image pushed to GHCR, deployed as RunPod community pod.

**Tech Stack:** NeMo (Parakeet), transformers 4.57.x (VibeVoice TTS + Qwen), FastAPI, Docker, runpodctl

**VRAM Budget:** Parakeet (~2GB) + VibeVoice TTS (~4GB) + Qwen-7B-Instruct (~15GB) = ~21GB of 24GB

---

### Task 1: Create Parakeet STT Service

**Files:**
- Create: `demo/conversation/parakeet_asr_service.py`

**Step 1: Create the Parakeet ASR service wrapper**

Same interface as existing ASRService but using NeMo Parakeet TDT 0.6B. Key differences: NeMo API, 16kHz input (resample from 24kHz), simpler text output.

```python
"""STT service using NVIDIA Parakeet TDT 0.6B via NeMo."""

import numpy as np
import tempfile
import soundfile as sf
from typing import Optional


class ParakeetASRService:
    """Wraps NVIDIA Parakeet TDT for real-time speech-to-text."""

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v3", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr
        print(f"[STT] Loading Parakeet from {self.model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[STT] Parakeet loaded.")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 24000,
    ) -> str:
        if self.model is None:
            raise RuntimeError("STT service not loaded")

        # Resample to 16kHz if needed (Parakeet expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        # NeMo expects file paths, so write to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio, 16000)
            output = self.model.transcribe([f.name])

        # Extract text from output
        if hasattr(output[0], 'text'):
            return output[0].text.strip()
        elif isinstance(output[0], str):
            return output[0].strip()
        return str(output[0]).strip()
```

---

### Task 2: Create Qwen LLM Response Generator

**Files:**
- Create: `demo/conversation/qwen_response_generator.py`

**Step 1: Create the Qwen response generator**

Loads Qwen2.5-7B-Instruct and generates conversational responses. Implements the existing ResponseGenerator interface.

```python
"""Response generator using Qwen2.5-7B-Instruct for conversation."""

import torch
from typing import List, Dict, Optional
from demo.conversation.response_generator import ResponseGenerator


class QwenResponseGenerator(ResponseGenerator):
    """Uses Qwen2.5-7B-Instruct for conversational responses."""

    SYSTEM_PROMPT = (
        "You are VibeVoice, a friendly and helpful voice AI assistant. "
        "Keep responses concise and conversational (1-3 sentences). "
        "You are speaking out loud, so avoid markdown, lists, or code blocks."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 150,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[LLM] Loading {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()
        print("[LLM] Qwen loaded.")

    def generate(self, transcription: str, history: List[Dict[str, str]]) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM service not loaded")

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        for entry in history[-10:]:
            role = entry.get("role", "user")
            messages.append({"role": role, "content": entry["text"]})
        messages.append({"role": "user", "content": transcription})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generated = output_ids[0, inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return response
```

---

### Task 3: Update app.py for Swappable STT/LLM

**Files:**
- Modify: `demo/conversation/app.py`

**Step 1: Update startup to support Parakeet + Qwen via env vars**

Add environment variable switches:
- `STT_ENGINE`: "parakeet" (default) or "vibevoice"
- `LLM_ENGINE`: "qwen" (default) or "template"
- `QWEN_MODEL`: model name for Qwen (default "Qwen/Qwen2.5-7B-Instruct")

Changes to `startup()`:
- If STT_ENGINE=parakeet, load ParakeetASRService instead of ASRService
- If LLM_ENGINE=qwen, load QwenResponseGenerator instead of TemplateResponseGenerator
- Call `.load()` on QwenResponseGenerator during startup

WebSocket handler stays the same - both STT services expose `.transcribe(audio, sample_rate)` and both response generators expose `.generate(text, history)`.

---

### Task 4: Create Dockerfile for RunPod

**Files:**
- Create: `Dockerfile.runpod`

**Step 1: Write the Dockerfile**

CUDA devel base for flash-attn. Installs NeMo + VibeVoice + Qwen. Downloads models at build time.

Key layers:
1. nvidia/cuda:12.4.0-devel-ubuntu22.04 + python3.11
2. PyTorch 2.5.1 + CUDA 12.4
3. flash-attn (needs CUDA devel headers)
4. nemo_toolkit[asr] (brings Parakeet support)
5. VibeVoice from GitHub (TTS deps)
6. Download models: parakeet-tdt-0.6b-v3, VibeVoice-Realtime-0.5B, Qwen2.5-7B-Instruct
7. Copy app code + voice presets
8. CMD uvicorn on port 8000

Notes:
- VibeVoice pip install URL needs the fork with exist_ok=True fixes
- Models baked into image to avoid download on every pod start
- Image will be ~15-20GB

---

### Task 5: Create RunPod Deployment Script

**Files:**
- Create: `scripts/runpod-deploy.sh`

**Step 1: Write deployment script using runpodctl**

```bash
#!/bin/bash
set -euo pipefail

IMAGE="ghcr.io/barockok/vibevoice-conversation:latest"
GPU_TYPE="NVIDIA GeForce RTX 3090"
CLOUD_TYPE="COMMUNITY"

runpodctl create pod \
  --name "vibevoice-conversation" \
  --gpuType "$GPU_TYPE" \
  --cloudType "$CLOUD_TYPE" \
  --imageName "$IMAGE" \
  --containerDiskSize 20 \
  --volumeSize 50 \
  --volumePath "/workspace" \
  --ports "8000/http" \
  --env "HF_HOME=/workspace/models" \
  --env "STT_ENGINE=parakeet" \
  --env "LLM_ENGINE=qwen"
```

---

### Task 6: Build, Push, Deploy

**Step 1:** Build Docker image: `docker build -f Dockerfile.runpod -t ghcr.io/barockok/vibevoice-conversation:latest .`
**Step 2:** Push to GHCR: `docker push ghcr.io/barockok/vibevoice-conversation:latest`
**Step 3:** Deploy: `bash scripts/runpod-deploy.sh`
**Step 4:** Verify: `runpodctl get pod` then open HTTP endpoint in browser

---

## Execution Order

1. Task 1 + Task 2 (parallel) - Create Parakeet STT + Qwen LLM services
2. Task 3 - Update app.py to wire everything together
3. Task 4 - Create Dockerfile
4. Task 5 - Create deploy script
5. Task 6 - Build, push, deploy
