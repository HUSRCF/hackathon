# ProtBind automatic quick-Vina selection results v3 — 2026-07-23

Record updated: 2026-07-24.

## Outcome

The current automatic-selection baseline is selection preparation 2.3, quick-input producer 1.1,
and `selection-quick-vina-1.2`. The direct AIAA-backed adapter smoke completed all three typed
requests and returned the complete 36-entry worker output closure. This direct protocol/environment
smoke remains `PASS`.

The current production attempt, `production-isolation-smoke-v3`, used a
`chemistry_verified=true` index, bound an explicit docking-box receipt through preparation and
worker input, completed through `SCREENED`, and reached the configured `SELECTED` bubblewrap
boundary. The host then denied loopback setup with `RTM_NEWADDR: Operation not permitted`; no worker
ran. Production OS isolation therefore remains `BLOCKED-BY-HOST`, separately from the successful
direct adapter smoke.

Quick Vina is selection-pruning evidence only. The later evidence-grade `DOCKED` stage must rerun
Vina and cannot promote a quick score as final docking evidence. v1 and v2 remain preserved as
historical records:

- [`EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA.md`](EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA.md)
- [`EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA_V2.md`](EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA_V2.md)
- [`EXPERIMENT_RESULTS_20260723_VINA_FIRST.md`](EXPERIMENT_RESULTS_20260723_VINA_FIRST.md)

## Frozen runtime and v3 profile

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
  `507fd3ac9d311cacd7df516e66d38043f46045d8727e6b36b58b489c8f742be9`.
- Evidence-grade/full-Vina worker code SHA-256:
  `e800f8b94c41a343582742d3a8bfbfacaa44a5fbde868f4bfcb58c7deb054334`.
- Shared local runtime-assets SHA-256:
  `e78b0d4eda4f223e7275270cdde325ae07cd86c490283319b87381853a0a0dd8`.
- Runtime trust remains `hash-attested-local-without-reviewed-upstream-allowlist`; the provenance
  receipt explicitly records `official_runtime=false`.

## Direct adapter smoke evidence

| Request | Molecule/microstate | Result | Vina tool score |
|---|---|---|---:|
| `quick-e8083ad652c8a182941ab51f` | `demo-001/state-01` | completed | -1.942 |
| `quick-3877ef605b5e431035ad62d3` | `demo-002/state-01` | completed | -1.896 |
| `quick-1c5646a433f6424b4d752dbb` | `demo-002/state-02` | completed | -1.753 |

Result coverage was 3/3 requests, with 36 worker output artifact references. Every score is labeled
“AutoDock Vina tool score only; not an experimental binding free energy.” The worker also emitted
the mandatory warning that quick Vina is selection-pruning evidence only and `DOCKED` must rerun
Vina.

Recorded aggregate timings were 1.614513140 s receptor preparation, 3.602890931 s ligand
preparation, 5.605610711 s Vina command total, 12.420815286 s quick-profile total, and
12.708330394 s process end to end. These are timings from one three-request CPU smoke run, not
throughput benchmarks.

## Docking-box receipt boundary

The direct run bound docking-box receipt SHA-256
`653249f4f516e4cde020ce82f38a963fa75cccf4800282f03e98d2cd72c18757`. It declares coordinate
frame `receptor-cartesian-angstrom` and records only geometry sanity checks: finite coordinates,
each dimension between 4 and 60 Å, and volume no greater than 27,000 Å³.

The same receipt explicitly records `receptor_atom_overlap_checked=false` and
`site_derivation_verified=false`. Therefore it prevents malformed geometry and missing
receptor/box identity bindings but does not establish that the box overlaps the receptor,
represents a biological pocket, or is suitable for docking science. The direct 1CRN box remains an
explicit public protocol-smoke box, not a known binding site.

## Production isolation workflow attempt

Run `production-isolation-smoke-v3` used
[`verified-smoke-index.sqlite`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/verified-smoke-index.sqlite),
whose metadata records `chemistry_verified=true`, SHA-256
`dda189cb5db528e7422a027340fbe643e4bae329493220fd789e50d7c628f8a7`.

The workflow produced and bound the following identities before attempting the worker:

| Object | SHA-256 |
|---|---|
| selection preparation 2.3 | `db5ed83970074bb3bc7a0223b207ef6c71470ef9c41de8ade439af2eb7d25e74` |
| quick-input producer 1.1 | `f596517eb5b25b2db58c31fb4d006ef806939ede09c353096abc3d672d3918f6` |
| production docking-box receipt | `641d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a` |

The production box receipt uses the same receptor Cartesian Å frame and the same sanity-only
limits. It also records `receptor_atom_overlap_checked=false` and
`site_derivation_verified=false`.

At the worker isolation boundary, bubblewrap exited before quick Vina ran:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

The workflow manifest records:

| Field | Recorded value |
|---|---|
| run | `production-isolation-smoke-v3` |
| state | `DEGRADED` |
| last completed stage | `SCREENED` |
| failed stage/code | `SELECTED` / `WORKER_CRASH` |
| recoverable | `true` |

The production manifest is
[`manifest.json`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/production-workspace/runs/production-isolation-smoke-v3/manifest.json),
SHA-256 `730e54551f807ed18896257fa3d2f47ba9da3a930c8bc2b7b8a2c6ebff585746`.

`protbind doctor` now preflights the same isolation capability and classifies this host as
`present_but_unusable` with exit status 1. The production runner never bypasses the failed
boundary. This proves the production protocol reaches, checks, and fails closed at its isolation
boundary; it does not prove successful OS isolation, worker execution, or candidate selection.
Isolation must not be weakened to make this smoke pass.

## Primary v3 receipts

| Artifact | SHA-256 | Interpretation |
|---|---|---|
| [`environment.json`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/environment.json) | `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35` | Path-redacted AIAA/core environment lock |
| [`vina-provenance.json`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/vina-provenance.json) | `b9b226eb718c2435f7450395f1ac40c2b1ae27a42ce40b02b8a316edfbae1536` | Full/quick Vina v3 identities and frozen CPU profile |
| [`smoke-result.json`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/smoke-result.json) | `c0e077c2d8c24e59fc4f6d3eece777f1c455b5fd325a7c890152d724339c11ee` | Three-request direct-adapter receipt, box binding, and output closure |
| [`direct box receipt`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/direct-workspace/objects/65/3249f4f516e4cde020ce82f38a963fa75cccf4800282f03e98d2cd72c18757) | `653249f4f516e4cde020ce82f38a963fa75cccf4800282f03e98d2cd72c18757` | Receptor-frame geometry sanity only; no overlap/site validation |
| [`verified-smoke-index.sqlite`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/verified-smoke-index.sqlite) | `dda189cb5db528e7422a027340fbe643e4bae329493220fd789e50d7c628f8a7` | Tiny chemistry-verified isolation-path fixture; not a scientific library |
| [`production box receipt`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/production-workspace/objects/64/1d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a) | `641d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a` | Bound by preparation/input; geometry sanity only |
| [`production manifest`](../experiment-results/aiaa-selection-quick-vina-20260723-v3/production-workspace/runs/production-isolation-smoke-v3/manifest.json) | `730e54551f807ed18896257fa3d2f47ba9da3a930c8bc2b7b8a2c6ebff585746` | Host-blocked, fail-closed production isolation attempt |

## Scientific and privacy boundaries

- The receptor is public 1CRN. Neither 1CRN nor either explicit box is an experimentally
  established protein-ligand test site in these runs.
- The successful direct adapter smoke used `demo-index.sqlite`, which has
  `chemistry_verified=false`; it is a synthetic workflow fixture, not a production screening
  library.
- The production attempt used `chemistry_verified=true`, but stopped at the isolation boundary
  before quick Vina ran. That flag proves only that the workflow chemistry gate accepted the tiny
  index; it does not establish scientific relevance or virtual-screening quality.
- No reference pose, PoseBusters gate, symmetry-aware RMSD, ProLIF IFP, OpenMM relaxation,
  enrichment, activity, affinity, or hit-rate evaluation was performed here.
- The successful 3/3 run is a direct adapter smoke under the application's offline policy, not a
  production-isolated execution. The production attempt remained fail-closed on the unusable host.
- The resulting poses, scores, ordering, timings, box receipts, and production failure therefore
  support protocol/environment behavior only. They do not support binding, ranking-quality,
  biological, virtual-screening, or Radeon-performance claims.

## Automated verification

- Full post-change pytest regression: 214 tests collected and passed, 100%, exit code 0.
- Ruff: `src`, `workers`, `tests`, and `scripts` passed.
- Compileall: passed.
- Focused contracts cover the automatic `SELECTED` path, box-receipt production and binding,
  coordinate-frame and geometry bounds, idempotent resume, no-box fail-closed behavior, cached
  all-failure behavior, CPU-only quick-profile enforcement, exact request coverage and reachable
  output closure, and rejection of quick input by the evidence-grade Vina worker.

These software checks do not remove the scientific, site-validity, or host-isolation limitations.

## Remaining gates

1. Re-run the same production path on a host/container where doctor reports the required
   bubblewrap network namespace and loopback setup as usable, keeping isolation unchanged and
   fail-closed.
2. Replace the protocol-smoke box with a scientifically justified pocket and independently verify
   receptor overlap/site derivation before interpreting any docking result.
3. Complete one public complex through `both`, `ligand_only`, and `pocket_only` to final
   Markdown/HTML reports, without requiring a cofolder.
4. Freeze the non-cherry-picked 10-complex regression and report PB-valid plus symmetry RMSD,
   ProLIF IFP recovery, strain, and complete failure/exclusion denominators.
5. Complete persisted-index TriPharm HIP top-512 parity/throughput and OpenMM HIP build gates.
