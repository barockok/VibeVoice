#!/bin/bash
set -euo pipefail

# VibeVoice Conversation Demo — RunPod Deployment
# Deploys a pod with CUDA base image, startup script installs everything.
# Models persist on /workspace volume across restarts.

GPU_TYPE="${GPU_TYPE:-NVIDIA GeForce RTX 3090}"
POD_NAME="${POD_NAME:-vibevoice-conversation}"
CONTAINER_DISK="${CONTAINER_DISK:-40}"
VOLUME_SIZE="${VOLUME_SIZE:-80}"

# Base CUDA image with Python
IMAGE="nvidia/cuda:12.4.0-devel-ubuntu22.04"

echo "=== VibeVoice Conversation Demo — RunPod Deploy ==="
echo "GPU:    $GPU_TYPE (COMMUNITY)"
echo "Image:  $IMAGE"
echo "Pod:    $POD_NAME"
echo "Disk:   ${CONTAINER_DISK}GB container + ${VOLUME_SIZE}GB volume"
echo ""

runpodctl create pod \
  --name "$POD_NAME" \
  --gpuType "$GPU_TYPE" \
  --communityCloud \
  --imageName "$IMAGE" \
  --containerDiskSize "$CONTAINER_DISK" \
  --volumeSize "$VOLUME_SIZE" \
  --volumePath "/workspace" \
  --ports "8000/http" \
  --startSSH \
  --env "HF_HOME=/workspace/models" \
  --env "STT_ENGINE=parakeet" \
  --env "LLM_ENGINE=qwen" \
  --env "MODEL_DEVICE=cuda"

echo ""
echo "Pod created. Next steps:"
echo "  1. runpodctl get pod                    # check status"
echo "  2. SSH into pod and run the startup script:"
echo "     curl -sSL https://raw.githubusercontent.com/barockok/VibeVoice/main/scripts/runpod-start.sh | bash"
echo ""
echo "  Or use RunPod web terminal to run the startup script."
