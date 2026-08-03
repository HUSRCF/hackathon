# ProtBind Pharmer / TriPharm CPU / TriPharm HIP three-way benchmark

## Status

The external Pharmer baseline is runnable without copying or linking GPL code into ProtBind.
The frozen baseline is the Pharmit backend from `docker.io/dkoes/pharmit:latest`, image digest
`sha256:3a44f86d00bdb1fc196dd8beb996602ecbf315551c4da30667d3705c9088cfb1`, source
revision `1904ebc2f5a9f886c5fd276d360545f9cc1c8a5d`. Pharmer remains an external
GPL-2.0-or-later process; ProtBind remains Apache-2.0.

The adapter preserves supplied DUD-E coordinates, generates deterministic ETKDGv3 conformers
for LIT-PCBA, freezes label-blind queries, retains failed molecules in the metric denominator,
hashes inputs and outputs, and checks exact TriPharm CPU/HIP ranked-ID parity. The implementation
is `src/protbind_agent/screening_benchmark.py`.

```bash
protbind benchmark pharmacophore-three-way \
  --dataset-name LIT-PCBA \
  --dataset-split "AVE-unbiased ESR1_ant validation" \
  --screen-labels labels.json \
  --index tripharm.sqlite \
  --query query.json \
  --pharmer-hit triangle-000.sdf \
  --pharmer-hit triangle-001.sdf \
  --pharmer-provenance pharmer-provenance.json \
  --backend hip \
  --hip-executable build/tripharm_hip/tripharm_hip_query \
  --output receipt.json
```

## Frozen runs on 2026-08-02

| Dataset / protocol | Engine | Hit-set recall | Hit-set precision | Hit-set EF | Result |
|---|---:|---:|---:|---:|---|
| DUD-E AMPC, supplied conformer, label-blind 3-point query | Pharmer | 0.000 | 0.000 | 0.000 | negative |
| DUD-E AMPC, same library/query | TriPharm CPU/HIP | 0.0484 | 0.1500 | 7.1710 | exploratory positive |
| DUD-E HS90A, supplied conformer, label-blind 3-point query | Pharmer | 0.0080 | 0.0588 | 2.3845 | modest enrichment |
| DUD-E HS90A, same library/query | TriPharm CPU/HIP | 0.2240 | 0.1143 | 4.6327 | exploratory positive |
| LIT-PCBA ESR1 antagonist validation, four conformers, 64-triangle panel | Pharmer | 0.000 | 0.000 | 0.000 | negative |
| LIT-PCBA ESR1 antagonist validation, same panel | TriPharm CPU/HIP | 0.0400 | 0.0088 | 0.2874 | below random prevalence |
| LIT-PCBA TP53 validation, exploratory single conformer/query | both | 0.000 | 0.000 | 0.000 | negative |

Every recorded TriPharm HIP run reproduced the complete CPU ranked molecule-ID order. These are
retrospective screening results, not experimental validation of new compounds. DUD-E is a biased
pilot set; LIT-PCBA is the stronger evidence and is currently negative for the tested
crystal-ligand-derived hypotheses.

Inputs came from the official [DUD-E target downloads](https://dude.docking.org/targets) and
[LIT-PCBA dataset](https://lab.drugdesign.unistra.fr/datasets/lit-pcba/). LIT-PCBA uses the
official AVE-unbiased validation split. Exact receipts are under
`experiment-results/pharmacophore-three-way-20260802/`.

## Performance boundary

The HIP kernel is fast and exact, but the current production adapter exports every SQLite
triangle for every call. It is therefore not yet an end-to-end speedup:

| DUD-E target | TriPharm CPU p50 | HIP end-to-end p50 | HIP kernel p50 |
|---|---:|---:|---:|
| AMPC, 563,902 triangles | 29.16 ms | 1.335 s | 38.16 us |
| HS90A, 2,584,816 triangles | 352.82 ms | 5.668 s | 95.99 us |

The supported claim is limited to a fast HIP triangle kernel with exact rank parity on gfx1100.
A production acceleration claim requires a persistent hash-bound binary export or resident GPU
index so SQLite export and process startup are amortized.

Pharmer wall time is not compared because it runs in an old container and its internal timer
reports zero for these small queries. It serves as an independent scientific baseline, not yet a
fair performance baseline.

## Frozen three-target prospective aggregate (2026-08-03)

The train-only protocol was frozen before validation under protocol SHA-256
`dbdaf33e0346f0ca2882d63d29495fd3851afd0effba0abf43b8c36b0c3f9864`. The one-shot
validation receipts retain every failed parent as a zero score and bind the complete CPU/HIP score
vectors. The aggregate receipt is
`experiment-results/pharmacophore-three-way-20260802/prospective-three-target-aggregate-v1.json`
with aggregate SHA-256
`5ac3a9caa8a17cd445ebb5422ee0df073da829df6ead1714bad40c443eaa362d`.

| Target | Records / actives | TriPharm AP lift | TriPharm EF1% | Pharmer AP lift | Pharmer EF1% | CPU/HIP |
|---|---:|---:|---:|---:|---:|---|
| ALDH1 | 26,640 / 1,343 | 1.072 | 0.966 | 1.056 | 1.560 | exact |
| MAPK1 | 15,327 / 77 | 1.759 | 1.293 | 1.517 | 1.293 | exact |
| MTORC1 | 8,267 / 24 | 2.665 | 4.150 | 0.861 | 0.000 | exact |

The median TriPharm AP is `0.0088355`, versus `0.0076193` for Pharmer, and the median TriPharm
EF1% is `1.2925`. All three targets contain at least one active in the top 1%, but only MAPK1 and
MTORC1 have both AP lift and EF1% above one. Therefore:

- protocol-complete gate: **PASS**;
- exploratory-positive gate: **PASS**;
- competition-strength gate: **FAIL**, because median EF1% is below the frozen threshold of 2;
- broad superiority over Pharmer: **not supported** (Pharmer has higher EF1% on ALDH1);
- CPU/HIP exactness: **PASS** on all three complete score vectors.

The acceleration claim also remains negative. On ALDH1, the TriPharm CPU full scan took
`12,021.27 s`; the Radeon kernel took `0.0667 s`, but the measured CPU exact finalizer in the HIP
path took `11,979.27 s`. MAPK1 and MTORC1 show the same pattern. These results support an exact,
fast Radeon prefilter kernel, not end-to-end application acceleration.

Five warm-ups followed by 30 measured static-index prefilter repetitions on GPU1
(W7900 Dual Slot, gfx1100) quantify that bounded kernel lane:

| Target | Frozen batch | Static index | Process p50 / p95 | Kernel p50 / p95 |
|---|---:|---:|---:|---:|
| MAPK1 | 4 queries | 144.5 MB | 0.2575 / 0.2692 s | 0.01057 / 0.01086 s |
| MTORC1 | 16 queries | 132.0 MB | 0.2794 / 0.2885 s | 0.03801 / 0.03829 s |
| ALDH1 | 16 queries | 252.5 MB | 0.3945 / 0.4041 s | 0.06638 / 0.06683 s |

These measurements include process startup, static-file load, transfers, the HIP prefilter and
result serialization, but deliberately exclude the CPU exact finalizer. Their receipts label this
boundary and cannot be cited as application-level speedup evidence.

## Scientific interpretation and next gate

Current evidence rejects a broad claim that one crystal-ligand pharmacophore reliably recovers
LIT-PCBA actives. The next protocol must be frozen before observing more validation results:

1. derive a small query ensemble only from `active_T.smi` and official co-crystal templates;
2. cluster or select medoids using training data only;
3. freeze query count, conformer count, tolerances, aggregation and ranking;
4. run untouched `active_V.smi` / `inactive_V.smi` once;
5. report EF1%, BEDROC or PR-AUC only for a complete ranking; otherwise report hit-set precision,
   recall and enrichment and mark incomplete top-fraction metrics;
6. keep Pharmer SMARTS versus RDKit BaseFeatures disagreement explicit rather than treating it as
   a GPU correctness failure.

The frozen ensemble now passes the exploratory scientific gate but not the competition-strength or
end-to-end acceleration gates. The competition-safe claim is an auditable local screening
workflow, modest target-dependent early enrichment, independent-baseline comparison, and exact
Radeon HIP parity—not validated biological recall, affinity prediction, broad superiority over
Pharmer, or end-to-end acceleration.
