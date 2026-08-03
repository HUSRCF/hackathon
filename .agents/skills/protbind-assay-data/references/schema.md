# Experimental assay schema 1.0

Accept CSV or TSV with these required columns:

| Column | Meaning |
|---|---|
| `experiment_id` | Stable experiment-series identifier |
| `assay_type` | `coip-western`, `cetsa`, `spr`, `bli`, `enzyme-activity`, or `cellular-response` |
| `target_id` | Target identifier used by the experiment |
| `candidate_id` | Compound or perturbation identifier |
| `batch_id` | Experimental batch |
| `lab_id` | Laboratory/site identifier |
| `condition_id` | Dose, time point, temperature, or other condition |
| `replicate` | Positive integer replicate number |
| `concentration` | Finite, non-negative numeric value |
| `concentration_unit` | One consistent unit per import |
| `response` | Finite measured response |
| `response_unit` | One consistent unit per import |
| `control_type` | `treatment`, `sample`, `none`, `vehicle`, `positive`, `negative`, or a documented control label |

One import must contain exactly one experiment, assay type, target, candidate, concentration unit,
and response unit. Batch, lab, condition, replicate, concentration, response, and control type may
vary by row.

The raw file and normalized JSON are immutable artifacts. SQLite stores the exact artifact
references and normalized measurement rows. Current operations are append-only; no arbitrary SQL,
replacement, or deletion surface is exposed.
