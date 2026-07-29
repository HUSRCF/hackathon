#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_env="${PROTBIND_AIAA_ENV:-AIAA}"
overlay_dir="${PROTBIND_AIAA_ESMFOLD_OVERLAY:-${repo_dir}/.venv-aiaa-esmfold-v1}"
checkout="${PROTBIND_ESMFOLD_OPENFOLD_CHECKOUT:-${repo_dir}/third_party/openfold-v1}"
revision="e938c184a291bf053af3b14c1e3e8bb29aee57e2"
gpu="${PROTBIND_ESMFOLD_GPU:-${HIP_VISIBLE_DEVICES:-0}}"

if [[ -n "${PROTBIND_ESMFOLD_GPU:-}" && -n "${HIP_VISIBLE_DEVICES:-}" && \
      "${PROTBIND_ESMFOLD_GPU}" != "${HIP_VISIBLE_DEVICES}" ]]; then
  printf 'PROTBIND_ESMFOLD_GPU and HIP_VISIBLE_DEVICES disagree\n' >&2
  exit 2
fi
if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ ]]; then
  printf 'PROTBIND_ESMFOLD_GPU must be one canonical numeric device\n' >&2
  exit 2
fi
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  printf 'HSA_OVERRIDE_GFX_VERSION is forbidden for competition evidence\n' >&2
  exit 2
fi
if [[ ! -x "${overlay_dir}/bin/python" ]]; then
  printf 'AIAA ESMFold v1 overlay is missing; run scripts/bootstrap-aiaa-esmfold-v1.sh first\n' >&2
  exit 1
fi
if [[ ! -d "${checkout}/.git" ]] || \
   [[ "$(git -C "$checkout" rev-parse HEAD)" != "$revision" ]] || \
   [[ -n "$(git -C "$checkout" status --porcelain)" ]]; then
  printf 'legacy OpenFold source is absent, modified, or not at the pinned revision\n' >&2
  exit 1
fi
if (($# == 0)); then
  printf 'usage: scripts/aiaa-esmfold-v1.sh COMMAND [ARG ...]\n' >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

exec conda run --no-capture-output -n "$base_env" /usr/bin/env \
  -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  "PATH=${overlay_dir}/bin:${repo_dir}/tools/bin:${PATH}" \
  "PYTHONPATH=${repo_dir}/src" \
  PYTHONNOUSERSITE="$PYTHONNOUSERSITE" \
  HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  HIP_VISIBLE_DEVICES="$gpu" \
  "$@"
