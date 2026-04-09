#!/bin/bash
set -euo pipefail

# VibeVoice Conversation Demo — RunPod Deployment
# Deploys Parakeet STT + Qwen LLM + VibeVoice TTS on a single GPU

IMAGE="${IMAGE:-ghcr.io/barockok/vibevoice-conversation:latest}"
GPU_TYPE="${GPU_TYPE:-NVIDIA GeForce RTX 3090}"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
POD_NAME="${POD_NAME:-vibevoice-conversation}"
CONTAINER_DISK="${CONTAINER_DISK:-20}"
VOLUME_SIZE="${VOLUME_SIZE:-50}"

echo "=== VibeVoice Conversation Demo — RunPod Deploy ==="
echo "Image:  $IMAGE"
echo "GPU:    $GPU_TYPE ($CLOUD_TYPE)"
echo "Pod:    $POD_NAME"
echo ""

runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "$GPU_TYPE" \
  --cloudType "$CLOUD_TYPE" \
  --imageName "$IMAGE" \
  --containerDiskSize "$CONTAINER_DISK" \
  --volumeSize "$VOLUME_SIZE" \
  --volumePath "/workspace" \
  --ports "8000/http" \
  --env "HF_HOME=/workspace/models" \
  --env "STT_ENGINE=parakeet" \
  --env "LLM_ENGINE=qwen" \
  --env "MODEL_DEVICE=cuda"

echo ""
echo "Pod created. Check status:"
echo "  runpodctl get pod"
