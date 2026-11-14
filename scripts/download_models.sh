#!/usr/bin/env bash
# Downloads model files that aren't fetched automatically.
#  - Kokoro TTS (ONNX, int8 quantised ~90MB; set KOKORO_FULL=1 for the fp32 ~310MB model)
#  - LiveKit turn-detector + Silero VAD weights (via the agent CLI)
# faster-whisper weights download from Hugging Face on first start of the speech server.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0
mkdir -p models/kokoro
MODEL=kokoro-v1.0.int8.onnx
[[ "${KOKORO_FULL:-0}" == "1" ]] && MODEL=kokoro-v1.0.onnx
for f in "$MODEL" voices-v1.0.bin; do
  if [[ ! -f "models/kokoro/$f" ]]; then
    echo "downloading $f"
    curl -fL --retry 3 -o "models/kokoro/$f.part" "$BASE/$f"
    mv "models/kokoro/$f.part" "models/kokoro/$f"
  fi
done

uv run python -m services.agent.main download-files
echo "done"
