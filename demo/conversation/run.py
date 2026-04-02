"""Entry point for the live conversation demo."""

import argparse
import os
import uvicorn


def main():
    p = argparse.ArgumentParser(description="VibeVoice Live Conversation Demo")
    p.add_argument("--port", type=int, default=3001)
    p.add_argument("--asr-model", type=str, default="microsoft/VibeVoice-ASR-0.5B")
    p.add_argument("--tts-model", type=str, default="microsoft/VibeVoice-Realtime-0.5B")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    p.add_argument("--voices-dir", type=str, default=None,
                   help="Path to voice presets dir (default: demo/voices/streaming_model)")
    args = p.parse_args()

    os.environ["ASR_MODEL_PATH"] = args.asr_model
    os.environ["TTS_MODEL_PATH"] = args.tts_model
    os.environ["MODEL_DEVICE"] = args.device
    if args.voices_dir:
        os.environ["VOICES_DIR"] = args.voices_dir

    uvicorn.run("demo.conversation.app:app", host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
