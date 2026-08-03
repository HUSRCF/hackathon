# TriPharm three-gate experiment tracker

Updated: 2026-08-03

| Run ID | Milestone | Purpose | System / variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| TG-M0-01 | M0 | freeze untouched target registry | protocol guard | ALDH1/MAPK1/MTORC1 | artifact hashes | MUST | PASS-CODE | exposed-target rejection, tamper rejection and atomic one-shot consumption tested; final protocol waits for query selection |
| TG-M0-02 | M0 | validate complete-score metrics | metric positive controls | synthetic | AP, EF1, BEDROC, ties | MUST | PASS | tie-invariant AP/ROC/BEDROC, EF boundary intervals and deterministic bootstrap tests pass |
| TG-M1-00 | M1 | query-bound cache pilot | TriPharm CPU/HIP | HS90A | p50 decomposition/parity | MUST | NEGATIVE | 5/5 exact; HIP p50 0.909 s versus CPU 0.359 s, speedup 0.396×; static batch/resident remains required |
| TG-M1-01 | M1 | static export round trip | CPU exporter | HS90A | corpus hash/counts | MUST | PASS | query-independent `TPHIPIDX1`, content-addressed cache, atomic write and tamper rejection; 2,584,816 triangles |
| TG-M1-02 | M1 | cached batch exactness | TriPharm CPU/HIP | HS90A | complete ranked-score hash | MUST | PASS-NEGATIVE-PERF | exact IDs and scores on gfx1100; warm end-to-end ~1.612 s versus CPU 0.804 s (0.50x); ensemble amortization remains to test |
| TG-M2-A | M2 | freeze target query ensemble | TriPharm train-only | ALDH1 T | AP then EF1 | MUST | PASS | selected K=16, tolerance=1.25 A; AP lift 1.145x, EF1 1.505; Pharmer panel hash-bound |
| TG-M2-M | M2 | freeze target query ensemble | TriPharm train-only | MAPK1 T | AP then EF1 | MUST | PASS | selected K=4, tolerance=1.00 A; AP lift 3.373x, EF1 3.980; Pharmer panel hash-bound |
| TG-M2-T | M2 | freeze target query ensemble | TriPharm train-only | MTORC1 T | AP then EF1 | MUST | PASS | selected K=16, tolerance=1.00 A; AP lift 23.118x, EF1 9.911; only 10 selection actives; Pharmer panel hash-bound |
| TG-M3-A-P | M3 | prospective external baseline | native Pharmer CPU | ALDH1 V | AP/EF1/BEDROC | MUST | PASS-WEAK | AP lift 1.056x, EF1 1.560; 751/751 outputs, missing/truncated 0 |
| TG-M3-A-C | M3 | prospective CPU result | TriPharm CPU | ALDH1 V | AP/EF1/BEDROC | MUST | PASS-WEAK | AP lift 1.072x, EF1 0.966, 13 top-1% actives; full 26,640 denominator; exact frozen recovery replay |
| TG-M3-A-H | M3 | prospective HIP result | TriPharm HIP | ALDH1 V | metrics + parity | MUST | PASS-PARITY-NEGATIVE-PERF | exact score hash; 0.0667 s kernel but 11,979.27 s CPU exact finalizer versus 12,021.27 s CPU |
| TG-M3-M-P | M3 | prospective external baseline | native Pharmer CPU | MAPK1 V | AP/EF1/BEDROC | MUST | PASS-WEAK | AP lift 1.517x, EF1 1.293, EF5 2.076 |
| TG-M3-M-C | M3 | prospective CPU result | TriPharm CPU | MAPK1 V | AP/EF1/BEDROC | MUST | PASS-WEAK | AP lift 1.759x, EF1 1.293, EF5 2.595; full 15,327 denominator |
| TG-M3-M-H | M3 | prospective HIP result | TriPharm HIP | MAPK1 V | metrics + parity | MUST | PASS-PARITY-NEGATIVE-PERF | exact score hash; ~508.25 s HIP path vs 505.17 s CPU (~0.994x) |
| TG-M3-T-P | M3 | prospective external baseline | native Pharmer CPU | MTORC1 V | AP/EF1/BEDROC | MUST | PASS-NEGATIVE | AP lift 0.861x, EF1 0.000; 812/812 outputs, missing/truncated 0 |
| TG-M3-T-C | M3 | prospective CPU result | TriPharm CPU | MTORC1 V | AP/EF1/BEDROC | MUST | PASS-EXPLORATORY | AP lift 2.665x, EF1 4.150, one top-1% active; full 8,267 denominator; wide CI from 24 actives |
| TG-M3-T-H | M3 | prospective HIP result | TriPharm HIP | MTORC1 V | metrics + parity | MUST | PASS-PARITY-NEGATIVE-PERF | exact score hash; 0.0378 s kernel but 1,657.40 s CPU exact finalizer versus 1,640.19 s CPU |
| TG-M4-01 | M4 | size break-even | TriPharm CPU/HIP | 1k/5k/25k/full | p50/p95/QPS | MUST | TODO | cold and warm separate |
| TG-M4-02 | M4 | ensemble throughput | TriPharm static HIP prefilter | three full LIT targets | p50/p95/QPS/parity | MUST | PASS-KERNEL-NEGATIVE-E2E | 5 warmups + 30 repeats: process p50 0.2575/0.2794/0.3945 s for MAPK1/MTORC1/ALDH1; formal receipts show CPU finalizer 506.57/1,657.40/11,979.27 s, so C2 FAIL |
| TG-M4-03 | M4 | cross-architecture exactness | W7900/R9700 | full selected target | ranked-score hashes | SHOULD | BLOCKED | R9700 not yet online |
| TG-M5-01 | M5 | bounded evidence package | all committed runs | three targets | claim matrix | MUST | PASS-DRAFT | aggregate `5ac3a9ca...362d`: gate complete PASS, exploratory PASS, competition-strength FAIL (median EF1 1.293); C2 negative preserved |
