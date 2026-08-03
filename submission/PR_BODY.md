# Track 2, <TEAM NAME>, ProtBind

## Application

ProtBind is a local-private protein–ligand research Agent running local Qwen inference through
HipFire on AMD Radeon GPUs. It separates LLM reasoning from deterministic scientific state and
supports auditable screening, docking, validation, local retrieval, optional sequence–SMILES
annotation, and immutable assay-data import.

## AMD Radeon / ROCm work

- Local Agent benchmark on Radeon Pro W7900 (`gfx1100`)
- ROCm/HIP TriPharm pharmacophore prefilter
- Exact CPU/HIP complete-score parity gates
- Runtime/model/source/hash-bound benchmark receipts

## Deliverables

- Source repository: <FINAL PUBLIC REPOSITORY URL>
- Project Specification PDF: <LINK OR PR PATH>
- Demo video: <PUBLIC VIDEO URL>
- PPTX: <LINK OR PR PATH>
- Reproducibility guide: <LINK OR PR PATH>

## Evidence boundaries

We claim local Radeon execution, successful audited Agent tool use, and an exact fast HIP
prefilter. We do not claim experimental affinity, broad superiority over Pharmer, prospective
docking generalization, or end-to-end screening acceleration.
