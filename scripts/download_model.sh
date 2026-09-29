#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ENV_HF="$ROOT/third_party/Speech/.venv/bin/hf"
[[ -x "$ENV_HF" ]] || { echo 'Run setup_env.sh first'; exit 1; }
"$ENV_HF" download nvidia/Nemotron-3-Diarization Nemotron-3-Diarization.nemo \
  --revision "${MODEL_REVISION:-main}" --local-dir checkpoints/nemotron_3_diarization
