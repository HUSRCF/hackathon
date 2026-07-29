# ProtBind hardened automatic quick-Vina selection results v2 — 2026-07-23

Record updated: 2026-07-24.

## Outcome

The current automatic-selection smoke baseline is now the hardened
`selection-quick-vina-1.1` adapter. A direct AIAA-backed run completed all three typed requests,
returned one evaluation per request, and exposed the complete 36-entry worker output closure.

Quick Vina remains selection-pruning evidence only. The later evidence-grade `DOCKED` stage must
rerun Vina; it cannot promote or reuse a quick score as final docking evidence. This v2 result
supersedes the one-request v1 smoke as the current implementation baseline, but the v1 receipts and
report remain preserved unchanged as historical evidence:

- [`EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA.md`](EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA.md)
- [`EXPERIMENT_RESULTS_20260723_VINA_FIRST.md`](EXPERIMENT_RESULTS_20260723_VINA_FIRST.md)

The direct v2 run closes only the hardened implementation/direct-adapter smoke gate. A subsequent
production workflow attempt reached the configured bubblewrap isolation boundary with a
chemistry-verified index, then failed closed because this host was not permitted to configure the
required loopback interface. Production isolation is therefore blocked by host OS capability; scientific
docking validation remains open.

## Frozen runtime and v2 profile

- Environment: AIAA plus the workspace overlay, Python 3.12.7, reusing AIAA Torch
  `2.12.1+rocm7.2`; no second Torch installation was introduced.
- Hardware audit: two AMD Radeon Pro W7900/gfx1100 devices, each reporting 48,301,604,864 bytes
  VRAM; `HSA_OVERRIDE_GFX_VERSION` was absent.
- Quick profile: AutoDock Vina 1.2.7, Meeko 0.7.1, RDKit 2025.9.3, Gemmi 0.7.5,
  NumPy 2.2.6, SciPy 1.16.0; `cpu=1`, `exhaustiveness=8`, `num_modes=1`, scoring `vina`.
- This adapter run was CPU-only. `peak_vram_bytes` is null, so it is not Radeon docking-performance
  evidence and must not be combined with GPU kernel timing.
- Environment-lock SHA-256:
  `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35`.
- Quick-worker code SHA-256:
  `a7ff6544aee91e58a27d015032eff1514a223bbb3d77d6a34abc88bc64747673`.
- Evidence-grade/full-Vina worker code SHA-256:
  `c7cfa990abf91edfaa644e642fc00bb19645a4daa8c11bb6eb420547dfff3f6b`.
- Shared local runtime-assets SHA-256:
  `e78b0d4eda4f223e7275270cdde325ae07cd86c490283319b87381853a0a0dd8`.
- Runtime trust remains `hash-attested-local-without-reviewed-upstream-allowlist`; the provenance
  receipt explicitly records `official_runtime=false`.

## Direct adapter smoke evidence

| Request | Molecule/microstate | Result | Vina tool score |
|---|---|---|---:|
| `quick-97ac17604d6a34d01e2e80a3` | `demo-001/state-01` | completed | -1.942 |
| `quick-873adfece52498634ca8bce6` | `demo-002/state-01` | completed | -1.896 |
| `quick-7d99ba4de392951c7a513f78` | `demo-002/state-02` | completed | -1.753 |

Result coverage was 3/3 requests, with 36 worker output artifact references. The score semantics in
every evaluation are exactly “AutoDock Vina tool score only; not an experimental binding free
energy.” The worker also emitted the mandatory warning that quick Vina is selection-pruning
evidence only and `DOCKED` must rerun Vina.

Recorded aggregate timings were 1.457 s receptor preparation, 3.362 s ligand preparation, 5.296 s
Vina command total, 11.621 s quick-profile total, and 11.845 s process end to end. These are timings
from one three-request smoke run, not throughput benchmarks.

Primary v2 receipts:

| Artifact | SHA-256 | Interpretation |
|---|---|---|
| [`environment.json`](../experiment-results/aiaa-selection-quick-vina-20260723-v2/environment.json) | `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35` | Path-redacted AIAA/core environment lock |
| [`vina-provenance.json`](../experiment-results/aiaa-selection-quick-vina-20260723-v2/vina-provenance.json) | `d1b17d7c43d034ee33310699780b79b47a03555a9b8f2a6dc979567342f7a0a5` | Full/quick Vina code identities and frozen v2 CPU profile |
| [`smoke-result.json`](../experiment-results/aiaa-selection-quick-vina-20260723-v2/smoke-result.json) | `962d6fe42b4a4c66f426d2d7c754817c2e51cf9084a07e7408886f242c25ee7a` | Three-request direct-adapter receipt and output closure |

## Production isolation workflow attempt

Run `production-isolation-smoke-v2c` used
[`verified-smoke-index.sqlite`](../experiment-results/aiaa-selection-quick-vina-20260723-v2/verified-smoke-index.sqlite),
whose metadata records `chemistry_verified=true`. Its SHA-256 is
`dda189cb5db528e7422a027340fbe643e4bae329493220fd789e50d7c628f8a7`.

The real workflow completed `INPUT_VALIDATED`, `RECEPTOR_READY`, `INDEXED`, and `SCREENED`, then
created the typed `SELECTED` worker input with SHA-256
`4b7126762c74582d5721b4c4040ee995393071f4d80bb6c5fd1a8bfd6ed81ac9`. At the production worker
isolation boundary, bubblewrap exited before quick Vina ran:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

The workflow manifest records:

| Field | Recorded value |
|---|---|
| run | `production-isolation-smoke-v2c` |
| state | `DEGRADED` |
| last completed stage | `SCREENED` |
| failed stage/code | `SELECTED` / `WORKER_CRASH` |
| recoverable | `true` |

The production manifest is
[`manifest.json`](../experiment-results/aiaa-selection-quick-vina-20260723-v2/production-workspace/runs/production-isolation-smoke-v2c/manifest.json),
SHA-256 `568449c52f98d0a0603704e528844292aa2475f2e6287fd68b0b4f771cd50c4a`.

This proves that the production workflow reaches and enforces the configured worker isolation
boundary and records a recoverable degraded state when the host denies it. It does not prove that
the production worker executed, that OS isolation succeeded, or that any candidate was selected.
The isolation policy must remain fail-closed; this result is not a reason to fall back to an
unisolated worker.

## Scientific and privacy boundaries

- The receptor is public 1CRN. Neither 1CRN nor the explicit box is an experimentally established
  protein-ligand test site in this run.
- The direct adapter smoke used `demo-index.sqlite`, which has `chemistry_verified=false`; it is a
  synthetic workflow fixture, not a production screening library.
- The separate production attempt used an index with `chemistry_verified=true`, but stopped at the
  isolation boundary before quick Vina ran. That flag proves only that the workflow chemistry gate
  accepted this tiny smoke index; it does not establish scientific relevance or virtual-screening
  quality.
- No reference pose, PoseBusters gate, symmetry-aware RMSD, ProLIF IFP, OpenMM relaxation,
  enrichment, activity, affinity, or hit-rate evaluation was performed here.
- The successful 3/3 smoke ran the adapter directly under the application's offline policy. The
  production attempt did invoke the bubblewrap boundary, but the host denied loopback setup, so OS
  network isolation has not yet succeeded. Production execution still requires that unchanged,
  fail-closed subprocess boundary on a capable host.
- The resulting poses, scores, ordering, and timings therefore support protocol/environment
  execution only. They do not support binding, ranking-quality, biological, or Radeon-performance
  claims.

## Automated verification

- Full post-change pytest regression: 203 tests collected and passed, 100%, exit code 0.
- Ruff: `src`, `workers`, `tests`, and `scripts` passed.
- Compileall: passed.
- Focused contracts cover the automatic `SELECTED` path, idempotent resume, no-box fail-closed
  behavior, cached all-failure behavior, CPU-only quick-profile enforcement, exact request coverage
  and reachable output closure, and rejection of quick input by the evidence-grade Vina worker.

These software checks do not remove the scientific and isolation limitations above.

## Remaining gates

1. Re-run the same production path on a host/container that permits the required bubblewrap
   network namespace and loopback setup, keeping the isolation policy unchanged and fail-closed.
2. Complete one public complex through `both`, `ligand_only`, and `pocket_only` to final
   Markdown/HTML reports, without requiring a cofolder.
3. Freeze the non-cherry-picked 10-complex regression and report PB-valid plus symmetry RMSD,
   ProLIF IFP recovery, strain, and complete failure/exclusion denominators.
4. Complete persisted-index TriPharm HIP top-512 parity/throughput and OpenMM HIP build gates.
