#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_env="${PROTBIND_AIAA_ENV:-AIAA}"
overlay_dir="${PROTBIND_AIAA_OPENFOLD_OVERLAY:-${repo_dir}/.venv-aiaa-openfold3}"
checkout="${PROTBIND_OPENFOLD_CHECKOUT:-${repo_dir}/third_party/openfold-3}"
revision="0bb17be5199846e806b6347b6e17c6249c88ff1b"
clone=false

if [[ "${1:-}" == "--clone" ]]; then
  clone=true
  shift
fi
if (($#)); then
  printf 'usage: scripts/bootstrap-aiaa-openfold3.sh [--clone]\n' >&2
  exit 2
fi
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  printf 'HSA_OVERRIDE_GFX_VERSION is forbidden for competition evidence\n' >&2
  exit 2
fi

if [[ ! -d "${checkout}/.git" ]]; then
  if [[ "$clone" != true ]]; then
    printf 'official OpenFold3 checkout is missing; re-run with --clone\n' >&2
    exit 1
  fi
  mkdir -p "$(dirname -- "$checkout")"
  git clone --depth 1 --branch 0.4.3 \
    https://github.com/aqlaboratory/openfold-3.git "$checkout"
fi

observed_revision="$(git -C "$checkout" rev-parse HEAD)"
if [[ "$observed_revision" != "$revision" ]]; then
  printf 'OpenFold3 revision mismatch: expected %s, got %s\n' \
    "$revision" "$observed_revision" >&2
  exit 1
fi

# This is the central competition optimization: inherit the already validated
# AIAA ROCm stack instead of allowing pip/pixi to download another Torch wheel.
conda run --no-capture-output -n "$base_env" python -c \
  'import sys, torch, triton; assert sys.version_info[:2] == (3, 12); assert torch.version.hip and torch.__version__.startswith("2.12."); assert triton.__version__ == "3.7.1"'

if [[ ! -x "${overlay_dir}/bin/python" ]]; then
  conda run --no-capture-output -n "$base_env" \
    python -m venv --system-site-packages "$overlay_dir"
fi

conda run --no-capture-output -n "$base_env" \
  "${overlay_dir}/bin/python" -m pip install \
  --upgrade-strategy only-if-needed \
  --constraint "${repo_dir}/requirements/aiaa-openfold3-overlay.lock.txt" \
  --requirement "${repo_dir}/requirements/aiaa-openfold3-overlay.txt"

conda run --no-capture-output -n "$base_env" \
  "${overlay_dir}/bin/python" -m pip install \
  --force-reinstall --no-build-isolation --no-deps --editable "$checkout"

"${repo_dir}/scripts/aiaa-openfold3.sh" validate-openfold3-rocm
"${repo_dir}/scripts/aiaa-openfold3.sh" python \
  "${repo_dir}/scripts/aiaa_openfold3_audit.py" \
  --output "${repo_dir}/experiment-results/aiaa-openfold3-environment.json"

printf '%s\n' \
  "AIAA-backed OpenFold3 runtime is ready." \
  "No checkpoint, CCD, private sequence, or duplicate Torch wheel was downloaded."
