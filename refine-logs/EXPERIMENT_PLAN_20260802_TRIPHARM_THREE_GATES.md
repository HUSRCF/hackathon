# TriPharm three-gate experiment plan

**Problem**: Establish whether TriPharm is a scientifically useful and reproducibly accelerated
pharmacophore-screening component, rather than only a correct HIP microkernel.

**Method thesis**: A train-only frozen pharmacophore ensemble, evaluated once on untouched
LIT-PCBA targets and executed through a content-addressed GPU index, can support bounded claims
about early enrichment, CPU/HIP correctness, and Radeon warm-query acceleration.

**Date**: 2026-08-02

## Claim map

| Claim | Why it matters | Minimum convincing evidence | Blocks |
|---|---|---|---|
| C1 — scientific retrieval | The Agent must retrieve experimental actives without adapting to validation labels. | A protocol frozen from training data only; full-population evaluation on three untouched targets; AP above prevalence and useful EF1% on at least two targets; all failures retained in the denominator; native Pharmer reported beside TriPharm. | B1, B2, B5 |
| C2 — Radeon acceleration | A fast kernel alone does not establish useful acceleration. | Exact CPU/HIP ranked-score parity; cached index identity; cold and warm timing separated; at least 5× warm or batched end-to-end speedup on HS90A and one LIT-PCBA-sized index, otherwise an explicit negative result. | B3, B4, B5 |

Anti-claims to rule out:

- enrichment is caused by DUD-E bias, a single co-crystal ligand, or validation-set tuning;
- Pharmer disagreement is incorrectly attributed to the HIP kernel despite different feature
  perception;
- speedup is obtained by excluding index export, process startup, CPU exact finalization, or failed
  molecules from the measured workload;
- stable-ID tie breaking creates apparent PR-AUC or EF gains among unmatched molecules.

## Scope and frozen target policy

The already observed ESR1 antagonist and TP53 validation results are development regressions only.
They cannot be promoted to prospective evidence.

The primary prospective target set is:

| Target | Train active / inactive | Validation active / inactive | Reason |
|---|---:|---:|---|
| ALDH1 | 4,020 / 76,577 | 1,343 / 25,297 | strongest statistical power with tractable validation size |
| MAPK1 | 231 / 46,317 | 77 / 15,250 | kinase target with moderate active count and library size |
| MTORC1 | 73 / 24,729 | 24 / 8,243 | smaller, feasible full-population cross-check |

PPARG is a contingency-only fourth target because its validation set contains only six actives.
ESR1_ant and TP53 remain useful for unit, integration, and failure-analysis development but are
excluded from the primary prospective aggregate.

Dataset identity must bind the official AVE-unbiased archive SHA-256
`1f50ef6bf66b8e987f056a2d2528f1d5a9031ad542ddc97f8ee2fbfd651c8de3` and the four split-file
hashes for every target. Split membership, record counts, parsing QC, and hashes may be inspected
to establish the denominator. Querying the validation library, calculating model scores or
retrieval metrics, or using any validation molecule to select a query or hyperparameter before the
protocol-freeze receipt is forbidden.

## Paper and competition storyline

Main evidence must prove:

1. train-only protocol freezing and one-shot validation on untouched targets;
2. native Pharmer versus TriPharm application-level retrieval with the feature-semantics caveat;
3. same-corpus TriPharm CPU/HIP exactness and honest end-to-end Radeon timing.

Supporting evidence can include DUD-E pilots, exposed ESR1/TP53 failures, query visualizations,
break-even curves, and W7900/R9700 cross-verification.

Experiments intentionally cut from the core story are affinity prediction, wet-lab hit validation,
docking-score comparison, a claim that Pharmer and RDKit perceive identical features, and a claim
that the current HIP prefilter performs the CPU exact-ranking stage on GPU.

## Block B1 — Sealed train-only query ensemble

- **Claim tested**: C1.
- **Why**: the present single-crystal query failed on LIT-PCBA and cannot be tuned on validation.
- **Inputs**: only each target's `active_T.smi`, `inactive_T.smi`, and public co-crystal templates.
- **Candidate generation**:
  - canonicalize parents before any split or clustering;
  - generate four deterministic ETKDGv3 candidate conformers per training active with a frozen seed;
  - map features through the current RDKit BaseFeatures definition;
  - discard fewer-than-three-feature candidates with an explicit failure receipt;
  - cluster training actives by Morgan radius-2/2048-bit Tanimoto and select deterministic medoids.
- **Inner development split**: hash parent identity into 80% fitting and 20% selection partitions.
  The inactive selection population is a hash-fixed sample of at most 10,000 records per target;
  all actives remain present. This subset is for protocol selection only, never final metrics.
- **Allowed grid**:
  - ensemble size: 4, 8, or 16 medoids;
  - distance tolerance: 0.75, 1.00, or 1.25 Å;
  - validation-library conformers: fixed at two unless a pre-validation resource gate forces one
    globally for all targets;
  - at most eight feature points and 64 high-information triangles per query.
- **Selection rule**: maximize inner-split average precision, then EF1%, then choose the smaller
  ensemble, then lower tolerance. No other manual selection is permitted.
- **Frozen aggregation**: a molecule score is the maximum geometric match score across ensemble
  queries; unmatched and preprocessing-failed records receive score zero and remain in the full
  denominator.
- **Output**: query JSON files, conformer and cluster receipts, chosen-grid receipt, source hashes,
  code hash, and a signed-off protocol manifest created before validation execution.
- **Success criterion**: deterministic byte-identical regeneration and no validation artifact in
  query provenance.
- **Failure interpretation**: inability to produce a stable ensemble blocks C1; it is not repaired
  by using a validation co-crystal or inspecting validation performance.
- **Priority**: MUST-RUN.

## Block B2 — One-shot prospective LIT-PCBA evaluation

- **Claim tested**: C1.
- **Systems**:
  1. native Pharmer CPU, external pinned GPL process;
  2. TriPharm CPU;
  3. TriPharm HIP prefilter plus the same CPU exact finalizer.
- **Application-level fairness**:
  - identical parent population and generated conformer SDF for all engines;
  - identical train-derived query ensemble and type mapping where representable;
  - identical failed-record denominator;
  - engine-native feature perception remains explicitly different and is reported as such.
- **Complete-score metrics**:
  - primary: average precision and AP/prevalence lift;
  - secondary: EF1%, BEDROC with alpha 20, ROC-AUC, hit recall and precision;
  - uncertainty: 1,000 target-stratified bootstrap replicates with a frozen seed;
  - unmatched molecules share score zero; threshold-based AP is used so arbitrary ID ordering inside
    the zero-score tie cannot improve the result.
- **Aggregate**: per-target results plus median across ALDH1, MAPK1, and MTORC1. Do not pool records
  across targets into a micro-average that lets ALDH1 dominate.
- **One-shot rule**: after the protocol-freeze hash is written, validation is executed once. Any
  later code repair creates a new protocol revision and cannot silently replace the original run.
- **Claim tiers**:
  - gate complete: all three targets attempted, all records accounted for, metrics and intervals
    emitted, and no leakage;
  - exploratory positive: AP lift greater than 1 and EF1% greater than 1 on at least two targets;
  - competition-strength positive: median EF1% at least 2, at least one active in the top 1% on two
    targets, and TriPharm not worse than native Pharmer on median AP;
  - otherwise: scientifically valid negative result.
- **Priority**: MUST-RUN.

## Block B3 — Fair baseline separation

- **Claims tested**: C1 and C2.
- **Lane A, application baseline**: native Pharmer searches its own index over the same SDF and
  train-derived query ensemble. Compare AP, EF1%, BEDROC, completeness, index build time, and query
  time. This answers whether the delivered workflow is competitive, not whether its kernel is
  algebraically identical.
- **Lane B, implementation baseline**: TriPharm CPU and HIP consume the exact same serialized
  triangle corpus, molecule table, tolerances, query triangles, and scoring contract. This lane
  supports correctness and acceleration claims.
- **Required receipts**: container digest/source revision, executable hashes, feature-definition
  hashes, SDF hash, triangle-corpus hash, query hash, complete ranked-score hash, and hardware/runtime
  identity.
- **Success criterion**: no table or prose combines Lane A disagreement with Lane B parity. A native
  Pharmer win is an acceptable result; an untraceable or mismatched population is not.
- **Priority**: MUST-RUN.

## Block B4 — Persistent HIP index and end-to-end benchmark

- **Claim tested**: C2.
- **Required implementation**:
  1. split the current per-call `TPHIPQ1` export into a static `TPHIPIDX1` corpus and a small query
     payload;
  2. store the static export under a content-addressed cache keyed by SQLite SHA-256, schema,
     TriPharm config, and exporter code identity;
  3. write atomically under a lock, verify header/count/hash before reuse, and fail closed on any
     mismatch;
  4. add a batch mode that loads the static corpus once and executes all ensemble queries;
  5. retain CPU exact finalization in the measured HIP path and label the backend
     `hip-prefilter+cpu-exact-ranking`;
  6. optionally add a resident worker only after batch mode is correct and benchmarked.
- **Correctness gate**: exact equality of complete molecule score/rank hashes between CPU and HIP,
  including zero-score molecules, on synthetic positive controls, HS90A, and all prospective targets.
- **Performance matrix**:
  - deterministic library prefixes near 1k, 5k, 25k, and full target size;
  - single query and frozen ensemble batch;
  - five warm-ups plus 30 measured warm repetitions;
  - five cold repetitions including file load and process startup;
  - index creation reported once and never hidden inside or outside a selectively chosen path;
  - report median, p95, queries/s, export size, host RSS, device memory, and break-even size.
- **Hardware**: GPU1 W7900/gfx1100 first; repeat exactness and the full-size performance point on
  R9700/gfx1201 when available. Use only `HIP_VISIBLE_DEVICES=1` on this host.
- **Pass criterion**: exact parity everywhere and at least 5× warm or batched end-to-end speedup over
  TriPharm CPU on HS90A plus one LIT-PCBA target. Cold-start may be slower but must be reported.
- **Failure interpretation**: parity failure blocks the HIP backend; speed failure retains HIP as a
  correctness-qualified experimental backend but forbids an acceleration claim.
- **Priority**: MUST-RUN.

## Block B5 — Failure analysis and bounded claims

- **Claims tested**: both, by trying to falsify them.
- **Diagnostics**:
  - per-target preprocessing failure rates by class without removing failures from denominators;
  - active/inactive score distributions and zero-score fractions;
  - Pharmer-only, TriPharm-only, and shared hits, separated from CPU/HIP parity;
  - query-medoid coverage and which query retrieves each active;
  - sensitivity to ensemble deletion using 4 versus the frozen choice, on training data and exposed
    development targets only;
  - performance time decomposition: cache lookup, static load, kernel, transfer, CPU finalization,
    and serialization.
- **Priority**: MUST-RUN for failure receipts; figures are NICE-TO-HAVE.

## Required code changes

| Component | Minimal change |
|---|---|
| `screening_benchmark.py` | complete score vectors, tie-safe AP/EF/BEDROC, bootstrap intervals, multi-query aggregation |
| new `screening_protocol.py` | target/split commitments, train-only query freeze, validation one-shot revision guard |
| new `pharmer_runner.py` | pinned external-process invocation, timeout, provenance, multi-query hit/score parsing |
| `tripharm_hip.py` | static content-addressed export, cache validation, batch request, detailed timing receipt |
| `kernels/tripharm_hip/tripharm_hip_query.cpp` | separate static index/query readers and batch execution with one static load |
| CLI | `pharmacophore-freeze`, `pharmacophore-evaluate`, and `tripharm-hip-index` commands |
| tests | metric positive controls, split leakage rejection, cache tampering, batch parity, timeout/failure closure |

## Run order and deadline milestones

| Milestone | Goal | Runs or implementation | Decision gate | Estimated cost | Main risk |
|---|---|---|---|---|---|
| M0 — Aug 2 | Seal scope and metrics | target manifest, split hashes, metric fixtures, exposed-target registry | validation command refuses to run without a frozen protocol | 4–6 CPU hours | accidentally treating exposed targets as prospective |
| M1 — Aug 2–3 | Remove HIP export bottleneck | static cache, batch binary, synthetic and HS90A parity | exact hashes and cache-tamper tests pass | 8–14 engineering hours; <1 GPU hour | binary format or stale-cache bug |
| M2 — Aug 3–4 | Freeze train-only ensembles | ALDH1, MAPK1, MTORC1 training pipeline and inner selection | deterministic query and protocol hashes | 12–24 CPU hours; 20–50 GB scratch | conformer/index explosion |
| M3 — Aug 4–5 | Run one-shot validation | Pharmer, CPU, HIP on all three full validation populations | all records attempted; no protocol mutation | 12–30 CPU hours; <2 GPU hours | ALDH1 wall time or negative science result |
| M4 — Aug 5 | Measure performance | 1k/5k/25k/full, cold/warm/batch, GPU1 | parity plus ≥5× pass or honest negative result | 2–4 GPU hours | CPU finalizer dominates |
| M5 — Aug 5–6 | Package evidence | tables, receipts, claim matrix, demo; optional R9700 cross-check | every slide claim points to a receipt | 6–10 hours | deadline compression |

## Compute and data budget

- W7900 GPU1: approximately 4–7 GPU hours including profiling and repeats.
- CPU preparation and Pharmer indexing: approximately 24–50 wall-clock hours, parallelized by
  target where memory permits.
- Scratch: reserve 50 GB; fail before generation if the free-space gate is not met.
- Validation conformers: two per molecule by default. If a dry-run projection exceeds 50 million
  triangles for any one index, switch all three targets to one conformer before the protocol is
  frozen; never reduce only the slow or unfavorable validation target afterward.
- Stochasticity: conformer and bootstrap seeds are fixed. Performance repetitions are repeated
  measurements, not model seeds.

## Stop/go rules

1. Do not screen or calculate retrieval metrics on prospective validation records before M0
   produces the protocol hash. Identity and denominator QC remain allowed and receipted.
2. Do not run full validation unless preprocessing accounts for at least 99% of parents; lower
   completeness must be fixed or explicitly accepted in a new frozen revision.
3. Stop the HIP path immediately on any CPU/HIP complete-score hash mismatch.
4. Do not retune after validation. A negative result remains the authoritative first run.
5. If time runs short, cut PPARG, resident-daemon work, and presentation figures first. Do not cut
   full denominators, Pharmer provenance, parity checks, or cold/warm separation.

## Final checklist

- [ ] ALDH1, MAPK1, and MTORC1 validation remain unscreened before protocol freeze
- [ ] query ensemble provenance contains training artifacts only
- [ ] complete-score metrics pass positive-control fixtures and tie tests
- [ ] native Pharmer and same-corpus CPU/HIP lanes are reported separately
- [ ] persistent index is content-addressed and tamper-evident
- [ ] CPU/HIP full ranked-score hashes match on every committed run
- [ ] cold, warm, batch, export, and CPU-finalization time are all visible
- [ ] all prospective records, including failures, remain in the denominator
- [ ] negative results cannot be overwritten by a tuned rerun
- [ ] final competition claims are derived from the receipts, not from DUD-E alone
