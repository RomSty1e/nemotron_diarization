#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CUDA_EXTRA="${1:-cu12}"
case "$CUDA_EXTRA" in cu12|cu13) ;; *) echo 'Usage: bash scripts/setup_env.sh cu12|cu13'; exit 2;; esac
command -v git >/dev/null || { echo 'Install git first'; exit 1; }
command -v uv >/dev/null || { echo 'Install uv first (README.md)'; exit 1; }
# Require this project to be its own Git root, not a directory in another repository.
if [[ ! -e .git ]]; then git init .; fi
if [[ ! -e third_party/Speech/.git ]]; then
  if git config -f .gitmodules --get submodule.third_party/Speech.url >/dev/null 2>&1; then
    git submodule update --init third_party/Speech
  else
    git submodule add https://github.com/NVIDIA-NeMo/Speech.git third_party/Speech
  fi
fi
if [[ -n "${NEMO_REVISION:-}" ]]; then
  if [[ -n "$(git -C third_party/Speech status --porcelain)" ]]; then
    echo 'Speech has local edits; refusing to change its revision.'; exit 1
  fi
  git -C third_party/Speech fetch origin "$NEMO_REVISION"
  git -C third_party/Speech checkout --detach FETCH_HEAD
fi
(
  cd third_party/Speech
  uv sync --locked --python 3.13 --extra asr --extra "$CUDA_EXTRA"
)
ENV_PY="$ROOT/third_party/Speech/.venv/bin/python"
uv pip install --python "$ENV_PY" -e "$ROOT"
git -C third_party/Speech rev-parse HEAD > third_party/Speech.commit
"$ENV_PY" -m scripts.check_env
printf '\nActivate: source third_party/Speech/.venv/bin/activate\n'
