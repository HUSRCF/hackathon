# ProtBind automatic quick-Vina selection results — 2026-07-23

## Outcome

ProtBind's `SELECTED` stage can now turn deterministic screening/scaffold/microstate requests into
typed quick-Vina work, validate exact request coverage and output closure, and retain the result for
idempotent resume. Quick Vina is restricted to selection pruning; the later evidence-grade
`DOCKED` stage must rerun Vina and cannot reuse the quick score as final docking evidence.

A direct AIAA-backed adapter smoke completed one real Meeko/AutoDock Vina request and returned one
completed evaluation plus the complete 18-artifact reachable output closure. This closes the
implementation and direct-adapter smoke gate. It does **not** close the production OS-isolation or
scientific docking-validation gates.

The preceding Vina-first/redocking report remains preserved as
[`EXPERIMENT_RESULTS_20260723_VINA_FIRST.md`](EXPERIMENT_RESULTS_20260723_VINA_FIRST.md).

## Frozen runtime and quick profile

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
  `b3f86bcd3d35272a426a6001ad7d1b1d6209a7d2f65552c1bf140f63f835d770`.
- Shared local runtime-assets SHA-256:
  `e78b0d4eda4f223e7275270cdde325ae07cd86c490283319b87381853a0a0dd8`.
- Runtime trust remains `hash-attested-local-without-reviewed-upstream-allowlist`; the provenance
  receipt explicitly records `official_runtime=false`.

## Direct adapter smoke evidence

| Request | Result | Vina tool score | Vina command | End to end | Output closure |
|---|---|---:|---:|---:|---:|
| `quick-052c503948864429a9afed60` | completed | -1.942 | 1.642 s | 5.895 s | 18 artifacts |

The score semantics in the receipt are exactly “AutoDock Vina tool score only; not an
experimental binding free energy.” The worker also emitted the mandatory warning that quick Vina
is selection-pruning evidence only and `DOCKED` must rerun Vina.

Recorded phase timings were 1.488 s receptor preparation, 1.241 s ligand preparation, 1.642 s
Vina command, 5.582 s quick-profile total, and 5.895 s process end to end. These are one-request
smoke timings, not throughput benchmarks.

Primary receipts:

| Artifact | SHA-256 | Interpretation |
|---|---|---|
| [`environment.json`](../experiment-results/aiaa-selection-quick-vina-20260723/environment.json) | `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35` | Path-redacted AIAA/core environment lock |
| [`vina-provenance.json`](../experiment-results/aiaa-selection-quick-vina-20260723/vina-provenance.json) | `7edb484b4f201a7b3826b99fc16ba5e1711887c464102e55931214dbe2e7c30d` | Full/quick Vina identities and frozen profile |
| [`smoke-result.json`](../experiment-results/aiaa-selection-quick-vina-20260723/smoke-result.json) | `1280f3f06a3504a6bac626f85eb1c8ba8ef42eba0252cad61cd13544db315c00` | Direct adapter smoke receipt and output closure |

## Scientific and privacy boundaries

- The receptor is public 1CRN. Neither 1CRN nor the explicit box is an experimentally established
  protein-ligand test site in this run.
- `demo-index.sqlite` has `chemistry_verified=false`; it is a synthetic workflow fixture, not a
  production screening library.
- No reference pose, PoseBusters gate, symmetry-aware RMSD, ProLIF IFP, OpenMM relaxation,
  enrichment, activity, affinity, or hit-rate evaluation was performed here.
- The smoke ran the adapter directly under the application's offline policy. It did not prove
  bubblewrap/OS network isolation. Production execution still requires the fail-closed bubblewrap
  subprocess boundary and a chemistry-verified input index.
- The resulting pose and score therefore support protocol/environment execution only. They do not
  support a binding, ranking-quality, or biological claim.

## Automated verification

- Full post-change pytest regression: 198 tests collected and passed, 100%, exit code 0.
- Ruff: `src`, `workers`, `tests`, and `scripts` passed.
- Compileall: passed.
- Focused contracts cover the automatic `SELECTED` path, idempotent resume, no-box fail-closed
  behavior, cached all-failure behavior, CPU-only quick-profile enforcement, exact coverage and
  reachable output closure, and rejection of quick input by the evidence-grade Vina worker.

These software checks do not remove the scientific and isolation limitations above.

## Remaining gates

1. Run the same automatic path through the production bubblewrap worker with a
   chemistry-verified mini-index and a scientifically justified receptor/box.
2. Complete one public complex through `both`, `ligand_only`, and `pocket_only` to final
   Markdown/HTML reports, without requiring a cofolder.
3. Freeze the non-cherry-picked 10-complex regression and report PB-valid plus symmetry RMSD,
   ProLIF IFP recovery, strain, and complete failure/exclusion denominators.
4. Complete persisted-index TriPharm HIP top-512 parity/throughput and OpenMM HIP build gates.
