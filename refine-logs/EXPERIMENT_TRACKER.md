# ProtBind experiment tracker

Updated: 2026-08-04

| ID | Gate | Status | Evidence |
|---|---|---|---|
| ENV-AIAA-01 | Core overlay reuses AIAA ROCm stack; no duplicate Torch | PASS | `experiment-results/aiaa-environment.json`; requirements exclude Torch |
| ENV-OF3-01 | Official OpenFold3 ROCm/Triton/Evoformer runtime validator | PASS | `experiment-results/aiaa-openfold3-environment.json` |
| ENV-OF3-02 | Real p2-155k checkpoint inference | PENDING | checkpoint intentionally absent; cofold remains optional |
| ROUTE-01 | RCSB and exact local cache intercept receptor folding | PASS | `aiaa-rcsb-network-smoke`, `aiaa-rcsb-offline-cache-smoke` |
| ROUTE-02 | ESMFold v1 receptor-only fallback contract | PASS | `esmfold-aiaa-attested-receipt.json`; 24-aa offline GPU smoke |
| FLOW-01 | `both`, `ligand_only`, `pocket_only` protocol through screening | PASS | three AIAA protocol smokes |
| FLOW-02 | Schema-2 `SELECTED → DOCKED → VALIDATED → REPORTED` without cofold | PASS | workflow/state/output tests; Vina is no longer downstream of `COFOLDED` |
| SELECT-01 | Automatic scaffold/microstate/quick-Vina selection and resumable result attachment | PASS | current v4: selection 2.5/input 1.2/quick 1.3/box receipt 2.0; direct Meeko/Vina smoke 3/3 with 36-entry closure, protein-atom overlap and explicit hypothesis-only semantics; fixture index/application-offline only |
| SELECT-02 | Production quick-Vina OS isolation with a chemistry-verified input | BLOCKED-BY-HOST | historical v3 verified-input workflow reached `SELECTED`; current doctor still reports bubblewrap `present_but_unusable`, rc 1, `RTM_NEWADDR`; no bypass is allowed |
| CAL-01 | Known-site calibration is consumed before candidate selection | PASS | real 1IEP PASS receipt binds canonical source redock, exact prepared receptor/prep receipt, target and box; selection 2.5 and quick input hashes recorded; native reference absent from quick input |
| HIP-01 | 100k triangle CPU/HIP exactness | PASS | recall 1.0; zero mask/float mismatches |
| HIP-02 | Persisted-index exactness and ≥5× end-to-end | PASS-PARITY / FAIL-SPEEDUP | complete CPU/HIP score-vector parity passes on ALDH1/MAPK1/MTORC1, but the CPU exact finalizer dominates full-path time; no end-to-end acceleration claim |
| HIP-03 | Static content-addressed HIP corpus and batch query mode | PASS-STATIC / PENDING-RESIDENT | five warm-ups + 30 measured static-index batch runs complete on three targets; process/kernel p50 are reported separately, while a long-lived resident service remains future work |
| PHARM-01 | Independent native Pharmer application baseline | PASS-PILOT | DUD-E and exposed ESR1/TP53 runs complete; feature semantics differ from RDKit and are reported separately |
| PHARM-02 | Complete-score, tie-safe AP/EF1/BEDROC metrics | PASS-CODE | positive controls, zero-score tie invariance and deterministic bootstrap pass |
| PHARM-03 | Train-only frozen query ensembles | PASS | ALDH1/MAPK1/MTORC1 query ensembles, tolerances and aggregation were selected from train-only inputs and hash-frozen before validation |
| PHARM-04 | One-shot untouched LIT-PCBA validation | PASS-PROTOCOL / FAIL-COMPETITION-STRENGTH | all three frozen validation runs and aggregate complete; exploratory-positive gate passes, median EF1% <2 competition-strength gate fails, and broad superiority over Pharmer is unsupported |
| AGENT-01 | Local Radeon Agent tool-use benchmark | PASS | W7900/gfx1100, HipFire/Qwen, 1 warm-up + 3 measured; tool sequence, success and artifact citation rates all 1.0; evidence-eligible receipt `protbind-agent-w7900-c58ca3c.json` |
| DRUTAI-01 | Optional isolated sequence-SMILES annotation | PASS-RUNTIME / ANNOTATION-ONLY | pinned `convmixer64` and real strict-confined Snap one-record smoke complete; no network interface, cache disabled, never binding evidence or hard filter |
| ASSAY-01 | Immutable experimental-table import and explicit curve fits | PASS-P0 | preview/commit, append-only catalog, 4PL and one-site fit substrate implemented; assay-specific images/kinetics/mixed-effects remain future work |
| DOCK-01 | Real Vina/Meeko/PoseBusters/sPyRMSD redocking | PASS | three final public runs; 2/3 top-1 and 3/3 top-5 recovery |
| DOCK-02 | Canonical SDF/PDBQT, receptor/pose receipts, source/code/tool/license lineage | PASS | final `result.json` receipts and contract tests |
| PRIV-01 | Native reference is validation-only and cannot affect committed docking input | PASS | three final runs; reference absent from argv/docking case; workflow noninterference tests |
| INPUT-01 | Unsupported chemistry/structure fails closed | PASS | 1UOU stereo, 1KE5 missing side chains, raw 1S3V sulfate; v1 retained the 7YZU Meeko failure, while v2 only accepts it after constrained geometry and real Meeko/RDKit validation |
| SCI-01 | Result-blind fixed-ten holdout, frozen inputs and complete attempted denominator | PASS | 308 source IDs, 298 explicit exclusions, 10 frozen; 10/10 attempted with no substitution; run plan binds holdout/input/code/tool/config hashes |
| SCI-02 | Fixed 10-complex PB/RMSD/IFP/strain regression | PASS | immutable original protocol remains 7/10; v1 is 8/10; frozen `repair-protocol-v2-restrained-sidechain` is 10/10 completed with independently recomputed top-1/top-5 9/10, IFP mean/median 0.6598/0.7014, 0 metric failures and `gate_complete=true`; v2 is controlled revision evidence, not prospective validation |
| PREP-01 | Conservative outside-pocket heavy-atom repair and constrained added-atom geometry | PASS | explicit revision/resume binding; 6 Å protected pocket, no loops/`--allow_bad_res`; v2 fixes all original heavy atoms, moves only added atoms/transient H, checks distances/chirality, and retries only narrow geometry/RDKit failures at 250→1000→5000; 7BTT/7YZU pass real Meeko at 250 |
| IFP-01 | Local, receipted ProLIF receptor preparation | PASS | 8 Å whole-residue union around both ligands; atom identity and coordinates preserved; used by original fixed-ten, remediation and `repair-protocol-v1` independent regressions |
| OMM-01 | OpenMM HIP platform and parameterized minimization parity | BLOCKED-BY-BUILD | current AIAA doctor exposes Reference/CPU only |
| QA-01 | Full automated regression | PASS | Ruff and compileall clean; 454 pytest tests pass in the pinned AIAA science overlay |

Next execution order:

1. Preserve the original fixed-ten, v1 and v2 unchanged. Freeze a new result-blind external set
   before evaluating v2 generalization; never relabel the observed fixed-ten revision as
   prospective evidence.
2. Install/integrate fpocket and P2Rank consensus; until then `ligand_only` must remain explicitly
   unavailable rather than guessing a whole-protein box.
3. Run one fixed public complex through `both`, `ligand_only`, and `pocket_only` from input to
   Markdown/HTML report, with no cofolder configured.
4. Integrate the implemented conservative repair, constrained optimizer and receipted ProLIF crop
   into the production validation path, then add explicit heterogen/cofactor retention and
   parameterization decisions.
5. Preserve the completed frozen TriPharm one-shot receipts, aggregate and 30-repeat static
   prefilter receipts unchanged. A resident-GPU exact finalizer is required before revisiting an
   application-speedup claim; repeat exactness on R9700/gfx1201 when that host is available.
6. Build/verify OpenMM HIP, then evaluate optional OpenFold3 only after a real checkpoint gate.
7. Re-run automatic quick Vina on a host that permits the bubblewrap network namespace, keeping
   production isolation unchanged and fail closed.
