#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
cd "$repo_root"

action=all
gpu=1
model=qwen3.6:27b
base_url=http://127.0.0.1:11435/v1
skip_agent=0
allow_dirty=0
output_rel=

usage() {
  sed -n '1,180p' "$script_dir/README.md"
}

while (($#)); do
  case "$1" in
    all|prepare|preflight|run)
      action=$1
      shift
      ;;
    --gpu)
      gpu=${2:?--gpu requires a numeric device index}
      shift 2
      ;;
    --model)
      model=${2:?--model requires a HipFire model name}
      shift 2
      ;;
    --base-url)
      base_url=${2:?--base-url requires a loopback /v1 URL}
      shift 2
      ;;
    --output)
      output_rel=${2:?--output requires a project-relative directory}
      shift 2
      ;;
    --skip-agent)
      skip_agent=1
      shift
      ;;
    --allow-dirty)
      allow_dirty=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
  echo "error: --gpu must be one canonical numeric device index" >&2
  exit 2
fi
case "$base_url" in
  http://127.0.0.1:[0-9]*/v1|http://localhost:[0-9]*/v1)
    ;;
  *)
    echo "error: --base-url must be an exact loopback HTTP URL ending in /v1" >&2
    exit 2
    ;;
esac

if [[ "$action" != prepare ]]; then
  if [[ -z "$output_rel" ]]; then
    run_label=$(date -u +%Y%m%dT%H%M%SZ)-$$
    output_rel="experiment-results/submission-demo/$run_label"
  fi
  case "$output_rel" in
    experiment-results/submission-demo/*)
      ;;
    *)
      echo "error: --output must stay under experiment-results/submission-demo/" >&2
      exit 2
      ;;
  esac
  if [[ -e "$output_rel" ]]; then
    echo "error: output already exists; choose a fresh --output directory" >&2
    exit 2
  fi
  mkdir -p "$output_rel"
fi

run_clean() {
  local -a clean_env=(
    env -i
    "HOME=$HOME"
    "PATH=$PATH"
    "LANG=${LANG:-C.UTF-8}"
    "TERM=${TERM:-xterm-256color}"
  )
  if [[ -n "${HIP_VISIBLE_DEVICES:-}" ]]; then
    clean_env+=("HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES")
  fi
  if [[ -n "${PROTBIND_AIAA_ENV:-}" ]]; then
    clean_env+=("PROTBIND_AIAA_ENV=$PROTBIND_AIAA_ENV")
  fi
  if [[ -n "${PROTBIND_AIAA_OVERLAY:-}" ]]; then
    clean_env+=("PROTBIND_AIAA_OVERLAY=$PROTBIND_AIAA_OVERLAY")
  fi
  "${clean_env[@]}" "$@"
}

python_entry=(scripts/aiaa-protbind.sh)
protbind=(scripts/aiaa-protbind.sh -m protbind_agent)
hip_executable=build/tripharm_hip/tripharm_hip_query

prepare() {
  echo "[prepare] Building the fixed ROCm/HIP demo binaries"
  cmake -S kernels/tripharm_hip -B build/tripharm_hip -DCMAKE_BUILD_TYPE=Release
  cmake --build build/tripharm_hip -j
  test -x "$hip_executable"
}

preflight() {
  echo "[preflight] Capturing path-redacted host and capability evidence"
  run_clean "${protbind[@]}" doctor >"$output_rel/doctor.json"
  run_clean "${python_entry[@]}" "$script_dir/verify_demo.py" preflight \
    --doctor "$output_rel/doctor.json" \
    --output "$output_rel/host-preflight.json"
  rocm-smi --showproductname --showmeminfo vram --showdriverversion

  if ((allow_dirty == 0)) && [[ -n "$(git status --short)" ]]; then
    echo "error: final recording requires a clean Git worktree; commit first or use --allow-dirty only for rehearsal" >&2
    exit 2
  fi
  git rev-parse HEAD >"$output_rel/source-commit.txt"

  if ((skip_agent == 0)); then
    echo "[preflight] Verifying the loopback HipFire model"
    run_clean "${python_entry[@]}" "$script_dir/check_hipfire.py" \
      --base-url "$base_url" --model "$model" \
      --output "$output_rel/hipfire-preflight.json"
  else
    echo "[preflight] Agent check skipped for deterministic rehearsal only"
  fi
}

run_demo() {
  echo "[run] Building a fresh deterministic synthetic TriPharm index"
  run_clean "${protbind[@]}" index build \
    --input examples/library.features.jsonl \
    --output "$output_rel/index.sqlite" \
    | tee "$output_rel/index-build.json"

  echo "[run] Measuring the CPU reference"
  run_clean "${protbind[@]}" benchmark \
    --index "$output_rel/index.sqlite" \
    --query examples/ligand-query.json \
    --output "$output_rel/tripharm-cpu.json" \
    --backend cpu --warmup-runs 2 --repetitions 5 --top-k 3

  echo "[run] Measuring the Radeon HIP prefilter with exact CPU finalization"
  HIP_VISIBLE_DEVICES=$gpu run_clean "${protbind[@]}" benchmark \
    --index "$output_rel/index.sqlite" \
    --query examples/ligand-query.json \
    --output "$output_rel/tripharm-hip.json" \
    --backend hip --hip-executable "$hip_executable" \
    --warmup-runs 2 --repetitions 5 --top-k 3

  run_clean "${python_entry[@]}" "$script_dir/verify_demo.py" parity \
    --cpu "$output_rel/tripharm-cpu.json" \
    --hip "$output_rel/tripharm-hip.json" \
    --output "$output_rel/cpu-hip-parity.json"

  echo "[run] Advancing the synthetic case through the HIP-gated SCREENED stage"
  HIP_VISIBLE_DEVICES=$gpu run_clean "${protbind[@]}" case run \
    --case submission/demo/case.json \
    --index "$output_rel/index.sqlite" \
    --run-id submission-demo-screened \
    --mode ligand_only \
    --stop-after screened \
    --worker-config submission/demo/screening.toml \
    --workspace "$output_rel/workspace" \
    | tee "$output_rel/case-run.json"

  run_clean "${python_entry[@]}" "$script_dir/verify_demo.py" case \
    --case-run "$output_rel/case-run.json" \
    --manifest "$output_rel/workspace/runs/submission-demo-screened/manifest.json" \
    --workspace "$output_rel/workspace" \
    --output "$output_rel/case-acceptance.json"

  if ((skip_agent == 0)); then
    echo "[run] Starting the local Agent approval demonstration"
    echo "Approve only after checking the displayed ShadowPlan."
    create_prompt="Create exactly one offline case using case_path '$output_rel/case.json', index_path '$output_rel/index.sqlite', and run_id 'submission-demo-agent'. Do not call any other tool. State that this is a synthetic protocol fixture, not scientific evidence."
    cp submission/demo/case.json "$output_rel/case.json"
    run_clean "${protbind[@]}" agent \
      --backend hipfire --model "$model" --base-url "$base_url" \
      --project-root . --workspace "$output_rel/agent-workspace" \
      --max-steps 6 --tool-routing --json "$create_prompt" \
      | tee "$output_rel/agent-create.json"

    status_prompt="Call case_status exactly once for run_id 'submission-demo-agent'. Do not call any other tool. Explain the current gate without claiming a scientific result."
    run_clean "${protbind[@]}" agent \
      --backend hipfire --model "$model" --base-url "$base_url" \
      --project-root . --workspace "$output_rel/agent-workspace" \
      --max-steps 6 --tool-routing --json "$status_prompt" \
      | tee "$output_rel/agent-status.json"
  else
    echo "[run] Agent approval demonstration skipped; this run is not a complete competition video run"
  fi

  echo "[complete] Demo artifacts: $output_rel"
  echo "Scientific boundary: all generated inputs are synthetic protocol fixtures."
}

case "$action" in
  prepare)
    prepare
    ;;
  preflight)
    preflight
    ;;
  run)
    test -x "$hip_executable"
    preflight
    run_demo
    ;;
  all)
    prepare
    preflight
    run_demo
    ;;
esac
