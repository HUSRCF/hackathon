#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
base_env="${PROTBIND_AIAA_ENV:-AIAA}"
overlay_dir="${PROTBIND_AIAA_OVERLAY:-${repo_dir}/.venv-aiaa-protbind}"
download_vina=false

usage() {
  printf '%s\n' \
    "usage: scripts/bootstrap-aiaa-protbind.sh [--download-vina]" \
    "" \
    "Build a small venv overlay that inherits the AIAA ROCm base environment." \
    "--download-vina fetches the official v1.2.7 x86_64 binary from GitHub."
}

while (($#)); do
  case "$1" in
    --download-vina)
      download_vina=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

command -v conda >/dev/null 2>&1 || {
  printf 'conda is required to access the %s base environment\n' "$base_env" >&2
  exit 1
}

conda run --no-capture-output -n "$base_env" python -c \
  'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

if [[ ! -x "${overlay_dir}/bin/python" ]]; then
  conda run --no-capture-output -n "$base_env" \
    python -m venv --system-site-packages "$overlay_dir"
fi

conda run --no-capture-output -n "$base_env" \
  "${overlay_dir}/bin/python" -m pip install \
  --upgrade-strategy only-if-needed \
  --constraint "${repo_dir}/requirements/aiaa-protbind-overlay.lock.txt" \
  --requirement "${repo_dir}/requirements/aiaa-protbind-overlay.txt"

conda run --no-capture-output -n "$base_env" \
  "${overlay_dir}/bin/python" -m pip install \
  --no-build-isolation --no-deps --editable "$repo_dir"

vina_path="${repo_dir}/tools/bin/vina"
vina_sha256="f31f774f723bba7bbe6e9d1c47577020eea9a8da16424284c043d22593570644"
vina_url="https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64"

if [[ "$download_vina" == true && ! -f "$vina_path" ]]; then
  mkdir -p "$(dirname -- "$vina_path")"
  curl -L --fail --show-error --output "${vina_path}.partial" "$vina_url"
  observed="$(sha256sum "${vina_path}.partial" | awk '{print $1}')"
  if [[ "$observed" != "$vina_sha256" ]]; then
    printf 'Vina SHA-256 mismatch: expected %s, got %s\n' \
      "$vina_sha256" "$observed" >&2
    exit 1
  fi
  mv "${vina_path}.partial" "$vina_path"
  chmod 0755 "$vina_path"
fi

if [[ ! -x "$vina_path" ]]; then
  printf '%s\n' \
    "Vina CLI is not installed. Re-run with --download-vina while network is available." >&2
  exit 1
fi

observed="$(sha256sum "$vina_path" | awk '{print $1}')"
if [[ "$observed" != "$vina_sha256" ]]; then
  printf 'Vina SHA-256 mismatch: expected %s, got %s\n' \
    "$vina_sha256" "$observed" >&2
  exit 1
fi

"${repo_dir}/scripts/aiaa-protbind.sh" \
  "${repo_dir}/scripts/aiaa_environment_audit.py" \
  --output "${repo_dir}/experiment-results/aiaa-environment.json"
printf 'ProtBind AIAA overlay is ready: %s\n' "$overlay_dir"
