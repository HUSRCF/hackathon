#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_env="${PROTBIND_AIAA_ENV:-AIAA}"
overlay_dir="${PROTBIND_AIAA_OVERLAY:-${repo_dir}/.venv-aiaa-protbind}"

if [[ ! -x "${overlay_dir}/bin/python" ]]; then
  printf '%s\n' \
    "ProtBind AIAA overlay is missing." \
    "Run scripts/bootstrap-aiaa-protbind.sh --download-vina first." >&2
  exit 1
fi

# FlagEmbedding otherwise discovers TensorFlow from AIAA; that combination is not a
# supported ProtBind path and has produced a native crash on this host.
export PYTHONNOUSERSITE=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export USE_TORCH=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec conda run --no-capture-output -n "$base_env" /usr/bin/env \
  "PATH=${repo_dir}/tools/bin:${overlay_dir}/bin:${PATH}" \
  PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
  USE_TF="$USE_TF" \
  TRANSFORMERS_NO_TF="$TRANSFORMERS_NO_TF" \
  USE_TORCH="$USE_TORCH" \
  TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
  HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  "${overlay_dir}/bin/python" "$@"
