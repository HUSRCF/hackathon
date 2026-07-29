#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec python3 -m radeon_agent.opencode_deepseek "$@"
