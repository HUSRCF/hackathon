# ProtBind AIAA implementation results — 2026-07-21

## Outcome

The AIAA base is compatible with the official OpenFold3 0.4.3 ROCm runtime. The production setup
now reuses AIAA's PyTorch `2.12.1+rocm7.2`, HIP `7.2.53211`, and Triton `3.7.1` through a dedicated
36 MiB overlay. The overlay contains zero local Torch distribution entries. The official
`validate-openfold3-rocm` checks passed for GPU visibility, HIP Triton, and the Evoformer kernel.

This is an environment result, not an OpenFold3 inference result: no OpenFold checkpoint has been
imported and no protein-ligand complex has been predicted in this run.

The runtime route is now:

```text
user receptor → exact local sequence cache → explicitly approved RCSB
→ legacy ESMFold v1 receptor prediction
→ top candidates only: gated OpenFold3, or gated ESMFold2
→ if no complex predictor passes: explicit degraded state; preserve tool evidence only
```

Legacy ESMFold v1 is receptor-only. It must never be reported as a ligand-pose/cofolding engine.

## Verified results

### Environment and regression

- AIAA Python: `3.12.7`.
- Radeon devices: two W7900/gfx1100 cards, each reporting `48,301,604,864` bytes VRAM.
- Core overlay: 587 MiB; OpenFold3 overlay: 36 MiB; ESMFold v1 overlay: 20 MiB.
- OpenFold3: official source version `0.4.3`, commit
  `0bb17be5199846e806b6347b6e17c6249c88ff1b`.
- Official OpenFold source allowlist: 317 files, canonical SHA-256
  `742e9bf654b13f67783d095a2327af3ed31163580eaa7b4c548e8a8eb2e68010`.
- Ruff: all checked source, worker, test, and script files passed.
- Pytest: 150 tests passed.

OpenFold3 production accepts only `openfold3-p2-155k` with the pinned expected size
`2,287,928,196` bytes. OpenFold3 0.4.3 does not expose a supported small/medium/large checkpoint
tier. One query uses one GPU; multiple cards distribute independent queries and do not pool VRAM.
The two-card default keeps GPU0 as the single scientific lane and GPU1 available for HipFire.

### Receptor-fold interception

An explicitly approved public 1CRN fetch sent only the PDB identifier to `files.rcsb.org` and
selected an exact-chain/QC-valid receptor without folding. A second offline run resolved the same
sequence from the local cache and made no network request.

- Network decision: `rcsb_imported`.
- Offline decision: `local_exact_sequence_cache`.
- Raw RCSB mmCIF SHA-256:
  `23787562c427d7c1abe5420e86d5f1d0a6c7007dec1e8ce85645a6d69c32e8ba`.
- Selected receptor SHA-256:
  `557a87ba4c43c38c8b245617d78f3c00a73100dafe451e59fcfb806482c2d366`.

### ESMFold v1 AIAA fallback

A 24-residue offline receptor prediction completed on GPU0 while GPU1 remained unused. The runtime
inherited AIAA's PyTorch `2.12.1+rocm7.2`; the ESMFold overlay contains no second Torch install.
The attested run loaded the model in 26.112 s, inferred in 3.653 s, completed end to end
in 37.425 s, and used `8,496,247,808` bytes (about 7.91 GiB) of peak allocated VRAM. The path-free
receipt contains two output artifacts and
no sequence or absolute home path.

The AIAA fair-esm 2.0.0 source predates Python 3.12's dataclass validation. ProtBind therefore
applies a fail-closed shim only when both official source hashes match; it changes two defaults to
`default_factory` and records the compatibility ID in runtime provenance. This run verifies the
local receptor-only execution path, not structure accuracy or ligand cofolding.

### Workflow modes

The `both`, `ligand_only`, and `pocket_only` protocol smokes reached `SCREENED` with no failure
records. They used the same deterministic three-molecule fixture index. Its
`chemistry_verified=false` marker means these runs verify workflow/provenance behavior only, not
chemical screening quality.

| Run | Final stage | Failures |
|---|---:|---:|
| `aiaa-both-smoke` | `SCREENED` | 0 |
| `aiaa-ligand-only-smoke` | `SCREENED` | 0 |
| `aiaa-pocket-only-smoke` | `SCREENED` | 0 |

### TriPharm-HIP microbenchmark

The gfx1100 triangle-matcher microbenchmark used 100,000 candidate triangles/molecule IDs, 64
query triangles, seed 20260721, and seven repetitions.

- CPU reference: 0.050214102 s.
- HIP kernel p50/p95: 0.000063760 / 0.000069685 s.
- H2D/D2H: 0.047288104 / 0.002991605 s.
- Allocated VRAM: 28,401,280 bytes.
- Match masks exact, recall 1.0, zero float-bit mismatches.

This is a kernel microbenchmark, not persisted-index top-512 or end-to-end 100k throughput. Kernel
time alone cannot be used to claim the final 5× target.

### Vina/PoseBusters environment smoke

The local AutoDock Vina 1.2.7 binary, Meeko 0.7.1, and PoseBusters 0.6.5 completed a deterministic
CPU smoke using ethanol against prepared 1CRN. Three modes were parsed; the Vina tool scores were
`-1.965`, `-1.723`, and `-1.640`, and the primary PoseBusters boolean gates were true.

This toy pair has no scientific ranking meaning. The scores are Vina pose-ranking tool scores,
not experimental binding free energies. The runtime trust level remains
`hash-attested-local-without-reviewed-upstream-allowlist`.

## Known gaps and stop conditions

- OpenFold3 checkpoint/CCD import and real low-memory inference have not run.
- ESMFold2 remains interface-only until the three-public-complex offline gfx1100 gate passes.
- The current normal production `COFOLDED` stage is still strict OpenFold3; when no complex
  predictor is configured, it stops in an explicit recoverable `DEGRADED` state. Continuing a
  docking-only report past that state needs a separately typed skip/pose-seed bundle and must not
  masquerade as `COFOLDED`.
- AIAA OpenMM exposes Reference, CPU, and OpenCL, but not HIP. No OpenMM HIP claim is allowed from
  this build.
- The three-mode runs and Vina run are protocol/environment smokes, not Astex/PoseBusters accuracy
  regression.
- The unused OpenFold `.pixi` runtime and workspace `pixi-cache` were explicitly deleted, freeing
  approximately 17.25 GiB. The official OpenFold ROCm/Triton/Evoformer validator still passed
  afterward, confirming that the AIAA-backed overlay does not depend on those directories. The
  small standalone `tools/bin/pixi` executable remains only as an ignored bootstrap/reference tool.
