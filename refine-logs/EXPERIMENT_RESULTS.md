# ProtBind current experiment results — 2026-07-27

Full evidence and case-level metrics:
[`EXPERIMENT_RESULTS_20260725_SCIENTIFIC_GAPS.md`](EXPERIMENT_RESULTS_20260725_SCIENTIFIC_GAPS.md).
Frozen repair revision and complete rerun:
[`EXPERIMENT_RESULTS_20260726_REPAIR_PROTOCOL_V1.md`](EXPERIMENT_RESULTS_20260726_REPAIR_PROTOCOL_V1.md).
Restrained-side-chain v2 freeze and complete rerun:
[`EXPERIMENT_RESULTS_20260727_RESTRAINED_SIDECHAIN_V2.md`](EXPERIMENT_RESULTS_20260727_RESTRAINED_SIDECHAIN_V2.md).

## Current outcome

- Result-blind fixed-ten PoseBusters holdout: 10 frozen and attempted, 8 completed, 2 failed closed,
  0 metric failures. PB-valid + symmetry-RMSD ≤2 Å is 7/10 at both top 1 and top-5 oracle;
  `gate_complete=false` because two receptors did not reach docking.
- Top-five recovery is recomputed per pose from each hash-bound multi-record Vina SDF; historical
  recovery booleans are consistency checks only.
- The eight completed cases include PoseBusters strain/clash and receipted ProLIF metrics using an
  8 Å whole-residue receptor crop; mean/median IFP Jaccard are 0.6248/0.6125.
- A separate retrospective repair ablation recovered 7BTT at 0.9270 Å but 7YZU still failed. It does
  not alter the frozen 7/10 formal result.
- The subsequently frozen `repair-protocol-v1` was run on all ten unchanged cases: 10 terminal,
  9 completed, 1 failed closed, and independently recomputed top-1/top-5 recovery both 8/10. IFP
  Jaccard mean/median are 0.6443/0.6250 across nine evaluated cases. The gate remains incomplete
  because 7YZU still fails receptor preparation.
- `repair-protocol-v2-restrained-sidechain` fixes every original heavy atom, optimizes only added
  side-chain atoms and transient hydrogens, checks nonbonded distance and chirality, then requires
  real Meeko/RDKit acceptance. On the same ten cases it completed 10/10 with 0 metric failures;
  independently recomputed top-1/top-5 are both 9/10, IFP Jaccard mean/median are
  0.6598/0.7014, and `gate_complete=true`.
- Known-site calibration is a real selection consumer: canonical source redock, exact prepared
  receptor/preparation receipt, target and box must all match before `both`-mode selection.
- Current v4 profile-1.3 / selection-2.5 / input-1.2 direct AIAA smoke completed 3/3 quick requests
  and the 36-entry output closure.

## Primary artifacts

| Artifact | SHA-256 |
|---|---|
| [`fixed-ten holdout`](../experiment-results/posebusters-redock-holdout-20260725/holdout.json) | `01d9fd57f31ef006601b6a1e982d2cf020d50761bc9c8f5bfe61497ccc064ca3` |
| [`fixed-ten run plan`](../experiment-results/posebusters-redock-fixed10-formal-20260725/run-plan.json) | `a3eb1736d63fb76086c111fddb56fff75793d307c76cc4f409a2e7fe266ff99c` |
| [`fixed-ten batch result`](../experiment-results/posebusters-redock-fixed10-formal-20260725/batch-result.json) | `89deee7806792f120ef23b6cf72aab8bd910c54ac6b929fbfa5b2dbd1a36ad06` |
| [`fixed-ten independent regression v2`](../experiment-results/posebusters-redock-fixed10-formal-20260725/regression-v2.json) | `cab37219c7918a852a35b0296199da8cc60e5bdb52c584f72a8d517e232a85a8` |
| [`repair remediation regression v2`](../experiment-results/posebusters-redock-repair-remediation-20260725/regression-v2.json) | `5fba720c25196b3584f4b03c32622f9beb11bcc30315ce7752e9d66f3c5de7aa` |
| [`repair-protocol-v1 run plan`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/run-plan.json) | `a488542db18d127e274f517b275557b34104046097419ba8519bb48ab06d2a7b` |
| [`repair-protocol-v1 batch result`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/batch-result.json) | `248a24f8d9ecabfdc7e2a5564853d5f830aa1c6a722cf8980a258d77133fb28a` |
| [`repair-protocol-v1 independent regression`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/independent-regression.json) | `e9c77eaf01583f421583971b626b1dcdba7a2e3ca94983bce3c3f935e7f4391c` |
| [`repair-protocol-v2 run plan`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/run-plan.json) | `c6d6b4117b8db4503ad535530195a70b0f439847c6e6bba86a3c0b21634c813c` |
| [`repair-protocol-v2 batch result`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/batch-result.json) | `43e413f46bea1396ee653f250c0e789b1022821354807ff14ecf91d14bec75a1` |
| [`repair-protocol-v2 independent regression`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/independent-regression.json) | `90e558fc93657b15dd5bf46862aa19e6d20e36cccef62d1520aa38ab6c9c8de1` |
| [`redock regression result`](../experiment-results/redock-regression-pilot-20260725/result.json) | `241c1dc039802b1445d595d00fa771269dff70602c61a17f70238b7552e0f27b` |
| [`pilot manifest`](../configs/redock-regression-pilot-20260723.json) | `9bcaa75605ff2687ce332c6206b449093283ad939f09021dd33ce3d10c0d5b8e` |
| [`v4 quick smoke`](../experiment-results/aiaa-selection-quick-vina-20260725-v4/smoke-result.json) | `8b19a1d9aa76af0f60dcf0d95c17aa86357598b315d40478dd59c238da0e6f8e` |
| [`v4 Vina provenance`](../experiment-results/aiaa-selection-quick-vina-20260725-v4/vina-provenance.json) | `b4f51a5b93e5ac676edb328d7f86b177c009c7e90a11b8a9e444d0f0e0c7739f` |
| [`1IEP calibration receipt`](../experiment-results/known-site-calibration-1iep-20260725/objects/65/0aff4a6910ec549f1b807f80d8c3ef424be9461818c5b5886cd6bfbffa4d82) | `650aff4a6910ec549f1b807f80d8c3ef424be9461818c5b5886cd6bfbffa4d82` |
| [`1IEP calibrated selection preparation`](../experiment-results/known-site-calibration-1iep-20260725/objects/48/9c594be042ff257046ada66dffbd69fb9659ace14589308a9a916f1a281d17) | `489c594be042ff257046ada66dffbd69fb9659ace14589308a9a916f1a281d17` |

## Claim boundary

The 7/10 result is the immutable original known-site-redocking baseline with failures retained.
The 8/10 v1 and 9/10 v2 results are separately frozen protocol revisions rerun over all ten cases;
because the same holdout had already been observed, they are controlled revision evidence, not new
prospective estimates or prospective hit rates. The two-case retrospective run cannot update any
of these frozen results.
Vina scores are not experimental free energies. PoseBusters energy fields are conformational
diagnostics. IFP agreement is not affinity evidence. The v4 1CRN run is a user-center,
unverified-chemistry, application-offline protocol smoke, not docking-quality or production
isolation evidence.

## Verification and blockers

- 273 pytest tests passed; Ruff and compileall passed after adding the constrained optimizer,
  narrow RDKit retry policy, protocol binding, and integration tests.
- Bundled Vina is detected and the two-gfx1100 policy preserves GPU 1 for other tools.
- fpocket and P2Rank are missing, so real `ligand_only` site discovery remains unavailable.
- OpenMM exposes CPU/Reference only; HIP validation remains blocked.
- Bubblewrap is present but loopback namespace setup is denied; production selection remains
  fail closed.
- Still required: a new result-blind external set to assess v2 generalization, one-complex
  three-mode final-report run, persisted-index TriPharm top-512 parity/performance, cofactor policy,
  and optional real OpenFold3 checkpoint evaluation.
