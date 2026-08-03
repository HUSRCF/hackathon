---
name: protbind-assay-data
description: Import, inspect, and fit private ProtBind wet-lab assay measurements through approval-gated, append-only tools. Use for CSV/TSV assay intake, dose-response or one-site curve fitting, experiment catalog inspection, batch/lab metadata, and evidence-bounded interpretation of Co-IP/Western, CETSA, SPR/BLI, enzyme, or cellular results.
---

# ProtBind Assay Data

Keep raw measurements immutable, make every database mutation explicit, and separate numerical fit
quality from biological interpretation.

## Import data

1. Require a reviewed project-local CSV or TSV matching
   [schema.md](references/schema.md).
2. Obtain fresh approval before reading private assay data.
3. Call `protbind_experiment_import_preview` first. Explain its row, batch, lab, control, unit, and
   source-hash summary without claiming an import occurred.
4. Call `protbind_experiment_import_commit` only after approval with the exact fresh `plan_id`.
5. Stop on a stale plan, conflicting units, non-finite measurements, duplicate experiment ID, or
   unsupported assay type.

Never edit an imported revision in place. Supersede and tombstone tools are not yet exposed; ask
the operator to preserve the source and wait for those controlled operations instead of modifying
the SQLite catalog or content-addressed artifacts directly.

## Inspect and fit

- Use `protbind_experiment_list` only after private-data approval. It returns metadata, not raw
  measurement values.
- Select the fit model from the experimental protocol before calling
  `protbind_experiment_fit_curve`.
- Use `four-parameter-logistic` for a declared dose-response design with positive concentrations.
- Use `one-site-binding` only when a one-site saturation model was selected in advance.
- Stop on insufficient concentrations, non-convergence, or parameter-identifiability warnings.

Never try several models and report only the best-looking result unless model comparison was
pre-registered. Cite the returned fit artifact and report that confidence intervals remain
`NOT_EVALUATED` when the receipt says so.

## Interpret evidence

A good curve fit demonstrates agreement between measurements and a declared numerical model. It
does not by itself prove direct binding, cellular target engagement, selectivity, causality, or
mechanism. Preserve assay-specific controls and require orthogonal evidence before upgrading a
scientific claim.

AI may format a draft table, identify missing metadata, choose a previously declared tool, and
explain a receipt. AI may not invent measurements, repair missing replicates, silently change
units, choose exclusions after seeing outcomes, or write/delete database rows without approval.
