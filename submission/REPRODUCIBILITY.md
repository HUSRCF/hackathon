# ProtBind reproducibility guide

## Supported reference host

- Linux x86_64
- AMD Radeon Pro W7900 (`gfx1100`)
- ROCm/HIP toolchain available through the host AIAA environment
- Python 3.12 in the validated AIAA overlay; package metadata supports Python 3.11+
- CMake 3.23+

The exact ROCm, Python-package, GPU, and runtime identities must be captured by `protbind doctor`
and the benchmark receipts. Do not infer reproducibility from a version string alone.

## Environment setup

The bootstrap script expects an existing Conda environment named `AIAA`. Override it with
`PROTBIND_AIAA_ENV` when necessary.

```bash
git clone <FINAL PUBLIC REPOSITORY URL>
cd hackathon
scripts/bootstrap-aiaa-protbind.sh --download-vina
```

The script creates `.venv-aiaa-protbind`, installs the pinned overlay, installs this repository in
editable mode, verifies the exact Vina SHA-256, and writes an environment audit receipt.

## Build the ROCm HIP programs

```bash
cmake -S kernels/tripharm_hip -B build/tripharm_hip -DCMAKE_BUILD_TYPE=Release
cmake --build build/tripharm_hip -j
```

Expected binaries:

- `build/tripharm_hip/tripharm_hip_benchmark`
- `build/tripharm_hip/tripharm_hip_query`
- `build/tripharm_hip/tripharm_hip_batch_query`

## Verify capabilities

```bash
scripts/aiaa-protbind.sh -m protbind_agent doctor
rocm-smi
```

Check the selected GPU architecture, runtime identities, optional scientific tools, and network
isolation status. Missing capabilities must remain explicit; do not bypass a failed gate.

## Run tests

```bash
.venv-aiaa-protbind/bin/python -m pytest
ruff check .
python -m compileall -q src tests
```

The current working tree collects 456 tests. The final submission must rerun this command after the
submission commit is frozen and record the resulting commit SHA.

## Minimal offline screening smoke

```bash
.venv-aiaa-protbind/bin/protbind index build \
  --input examples/library.features.jsonl \
  --output artifacts/protbind/library.sqlite

.venv-aiaa-protbind/bin/protbind case run \
  --case submission/demo/case.json \
  --index artifacts/protbind/library.sqlite \
  --mode ligand_only \
  --stop-after screened
```

This example is a protocol smoke, not a biological validation result.

For the competition recording, use the stricter one-command workflow. It requires a real Radeon
HIP execution, complete CPU/HIP ranked-ID parity, a fresh synthetic workspace, and an interactive
Agent approval:

```bash
submission/demo/run-demo.sh run --gpu 1 --model qwen3.6:27b
```

## Start the local Agent

Start HipFire on a loopback-only endpoint with the exact local Qwen model intended for the demo.
Then run:

```bash
scripts/aiaa-protbind.sh -m protbind_agent agent \
  --backend hipfire \
  --model qwen3.5:9b \
  --workspace artifacts/protbind \
  --project-root . \
  "Inspect the case status and explain the next safe action."
```

Mutating or private-data calls pause for host approval. The model cannot approve itself.

## Evidence reproduction boundaries

The large benchmark datasets and third-party model weights are not redistributed automatically.
Their receipts bind source, license policy, and hashes. Reproduction should distinguish:

- code-path smoke tests;
- Radeon runtime and performance measurements;
- retrospective scientific benchmarks;
- controlled protocol revisions;
- future prospective biological validation.

Never report a Vina score as affinity, a DrutAI class as binding evidence, or a HIP kernel timing as
end-to-end application speedup.
