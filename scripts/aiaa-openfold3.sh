#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_env="${PROTBIND_AIAA_ENV:-AIAA}"
overlay_dir="${PROTBIND_AIAA_OPENFOLD_OVERLAY:-${repo_dir}/.venv-aiaa-openfold3}"
gpu="${PROTBIND_OPENFOLD_GPU:-${HIP_VISIBLE_DEVICES:-0}}"

if [[ -n "${PROTBIND_OPENFOLD_GPU:-}" && -n "${HIP_VISIBLE_DEVICES:-}" && \
      "${PROTBIND_OPENFOLD_GPU}" != "${HIP_VISIBLE_DEVICES}" ]]; then
  printf 'PROTBIND_OPENFOLD_GPU and HIP_VISIBLE_DEVICES disagree\n' >&2
  exit 2
fi

if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'PROTBIND_OPENFOLD_GPU must be one canonical numeric device\n' >&2
  exit 2
fi
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  printf 'HSA_OVERRIDE_GFX_VERSION is forbidden for competition evidence\n' >&2
  exit 2
fi
if [[ ! -x "${overlay_dir}/bin/python" ]]; then
  printf '%s\n' \
    "AIAA OpenFold3 overlay is missing." \
    "Run scripts/bootstrap-aiaa-openfold3.sh --clone first." >&2
  exit 1
fi
if (($# == 0)); then
  printf 'usage: scripts/aiaa-openfold3.sh COMMAND [ARG ...]\n' >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec conda run --no-capture-output -n "$base_env" /usr/bin/env \
  -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  "PATH=${overlay_dir}/bin:${repo_dir}/tools/bin:${PATH}" \
  PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
  HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  HIP_VISIBLE_DEVICES="$gpu" \
  "$@"
