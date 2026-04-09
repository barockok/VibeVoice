#!/bin/bash
set -euo pipefail

# VibeVoice Conversation Demo — RunPod Startup Script
# Runs inside the pod: installs deps, downloads models, starts server.
# Models are cached on /workspace/models (persistent volume).

export HF_HOME=/workspace/models
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

REPO_DIR=/workspace/VibeVoice
BRANCH="${BRANCH:-feat/live-conversation-demo}"

echo "=== VibeVoice Conversation Demo — Pod Startup ==="
echo "Branch: $BRANCH"

# --- System deps ---
apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

# --- Clone or update repo ---
if [ -d "$REPO_DIR/.git" ]; then
    echo "[setup] Updating repo..."
    cd "$REPO_DIR"
    git fetch origin && git checkout "$BRANCH" && git pull origin "$BRANCH"
else
    echo "[setup] Cloning repo..."
    git clone --branch "$BRANCH" --single-branch \
        https://github.com/barockok/VibeVoice.git "$REPO_DIR"
    cd "$REPO_DIR"
fi

# --- Install Python deps (skip if marker exists) ---
MARKER="/workspace/.deps-installed"
if [ ! -f "$MARKER" ]; then
    echo "[setup] Installing Python dependencies..."
    pip install --upgrade pip setuptools wheel

    # Upgrade PyTorch to 2.6.0+cu124 (NeMo 2.7.2 requires >=2.6.0)
    # Use --index-url (not --extra-index-url) to force cu124 wheels
    pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

    # Pin transformers to 4.x (5.x breaks VibeVoice internals)
    pip install "transformers>=4.57.0,<5.0.0"

    # Flash attention
    pip install ninja packaging
    pip install flash-attn --no-build-isolation

    # NeMo for Parakeet STT
    pip install "nemo_toolkit[asr]"

    # Re-pin after NeMo (it may pull incompatible versions)
    pip install "transformers>=4.57.0,<5.0.0"
    pip install torch==2.6.0 torchvision torchaudio \
        --extra-index-url https://download.pytorch.org/whl/cu124

    # VibeVoice from local checkout (with exist_ok fixes)
    pip install -e "$REPO_DIR"

    # Extra deps
    pip install hf_transfer soundfile

    touch "$MARKER"
    echo "[setup] Dependencies installed."
else
    echo "[setup] Dependencies already installed, skipping."
    pip install -e "$REPO_DIR" --quiet
fi

# --- Download models (cached on volume) ---
echo "[setup] Ensuring models are downloaded..."
python3 -c "
from huggingface_hub import snapshot_download
import os

models = [
    'nvidia/parakeet-tdt-0.6b-v3',
    'microsoft/VibeVoice-Realtime-0.5B',
    'Qwen/Qwen2.5-7B-Instruct',
]
for m in models:
    print(f'[models] Downloading {m}...')
    snapshot_download(m)
    print(f'[models] {m} ready.')
"

# --- Copy voice presets if not on volume ---
VOICES_DIR=/workspace/voices/streaming_model
if [ ! -d "$VOICES_DIR" ] || [ -z "$(ls -A $VOICES_DIR 2>/dev/null)" ]; then
    echo "[setup] Copying voice presets..."
    mkdir -p "$VOICES_DIR"
    cp -r "$REPO_DIR/demo/voices/streaming_model/"* "$VOICES_DIR/"
fi

# --- Start server ---
echo "[startup] Starting VibeVoice conversation server on port 8000..."
cd "$REPO_DIR"

export STT_ENGINE="${STT_ENGINE:-parakeet}"
export LLM_ENGINE="${LLM_ENGINE:-qwen}"
export TTS_MODEL_PATH="${TTS_MODEL_PATH:-microsoft/VibeVoice-Realtime-0.5B}"
export VOICES_DIR="$VOICES_DIR"
export MODEL_DEVICE=cuda

exec python3 -m uvicorn demo.conversation.app:app --host 0.0.0.0 --port 8000
