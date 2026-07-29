#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_env="${PROTBIND_AIAA_ENV:-AIAA}"
overlay_dir="${PROTBIND_AIAA_ESMFOLD_OVERLAY:-${repo_dir}/.venv-aiaa-esmfold-v1}"
checkout="${PROTBIND_ESMFOLD_OPENFOLD_CHECKOUT:-${repo_dir}/third_party/openfold-v1}"
revision="e938c184a291bf053af3b14c1e3e8bb29aee57e2"
clone=false

if [[ "${1:-}" == "--clone" ]]; then
  clone=true
  shift
fi
if (($#)); then
  printf 'usage: scripts/bootstrap-aiaa-esmfold-v1.sh [--clone]\n' >&2
  exit 2
fi
if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  printf 'HSA_OVERRIDE_GFX_VERSION is forbidden for competition evidence\n' >&2
  exit 2
fi

if [[ ! -d "${checkout}/.git" ]]; then
  if [[ "$clone" != true ]]; then
    printf 'official legacy OpenFold checkout is missing; re-run with --clone\n' >&2
    exit 1
  fi
  mkdir -p "$(dirname -- "$checkout")"
  git clone https://github.com/aqlaboratory/openfold.git "$checkout"
fi
git -C "$checkout" checkout --detach "$revision"
if [[ -n "$(git -C "$checkout" status --porcelain)" ]]; then
  printf 'legacy OpenFold checkout contains unreviewed modifications\n' >&2
  exit 1
fi

conda run --no-capture-output -n "$base_env" python -c \
  'import sys, torch, esm, omegaconf; assert sys.version_info[:2] == (3, 12); assert torch.version.hip and torch.__version__.startswith("2.12.")'

if [[ ! -x "${overlay_dir}/bin/python" ]]; then
  conda run --no-capture-output -n "$base_env" \
    python -m venv --system-site-packages "$overlay_dir"
fi

conda run --no-capture-output -n "$base_env" \
  "${overlay_dir}/bin/python" -m pip install \
  --upgrade-strategy only-if-needed \
  --constraint "${repo_dir}/requirements/aiaa-esmfold-v1-overlay.lock.txt" \
  --requirement "${repo_dir}/requirements/aiaa-esmfold-v1-overlay.txt"

# Build only the legacy OpenFold package and its CPU attention stub against
# AIAA's existing Torch. --no-deps is the hard guard against a second Torch.
conda run --no-capture-output -n "$base_env" \
  "${overlay_dir}/bin/python" -m pip install \
  --force-reinstall --no-deps --no-build-isolation "$checkout"

conda run --no-capture-output -n "$base_env" \
  "${overlay_dir}/bin/python" -c \
  'import pathlib, site, sys; local=next(pathlib.Path(p) for p in site.getsitepackages() if pathlib.Path(p).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())); (local / "protbind_legacy_openfold.pth").unlink(missing_ok=True)'

"${repo_dir}/scripts/aiaa-esmfold-v1.sh" python -c \
  'import importlib.metadata as m, torch, tree; from protbind_agent.esmfold_compat import install_fair_esm_py312_compat; install_fair_esm_py312_compat(); from esm.esmfold.v1.pretrained import _load_model; assert m.version("dm-tree") == "0.1.10"; assert m.version("ml-collections") == "1.0.0"; assert m.version("modelcif") == "0.7"; assert m.version("openfold") == "2.2.0"; assert torch.version.hip'

printf '%s\n' \
  "AIAA-backed ESMFold v1 runtime is ready." \
  "No checkpoint or duplicate Torch wheel was downloaded."
