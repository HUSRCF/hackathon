# Scientific boundaries

## Interpret the workflow honestly

- Prefer a user-supplied or explicitly approved RCSB structure over folding when it matches the
  target and passes coordinate/sequence quality control.
- Treat a public-download receipt as acquisition provenance only. Gemmi missing-backbone,
  unobserved-residue, altloc, and occupancy observations are deterministic triage, not proof that
  the selected chains are the intended receptor or that the structure is docking-ready.
- Treat PROPKA output as pKa/protonation feasibility plus tool diagnostics. PROPKA success does not
  prove residue/atom completeness, repair missing loops, validate a binding site, or accept a
  receptor; normal identity, chemistry, covalency, metal, and receptor-quality gates remain.
- Use ESMFold v1 only to obtain a receptor structure from protein sequence. It does not perform
  protein-ligand cofolding and supplies no ligand pose.
- Treat OpenFold3, when separately configured, as optional protein-small-molecule pose evidence.
  Its confidence is model confidence, not binding affinity or observed binding.
- Treat TriPharm as a geometric pharmacophore screen. Its score is not an affinity.
- Treat P2Rank output as a ranked binding-site hypothesis. A predicted pocket or calibrated model
  probability is not an observed site and must still pass coordinate, box, and docking gates.
- Treat DrutAI, if a separately pinned worker later passes its compatibility/license/model-card
  bake-off, as an annotation-only protein/SMILES classifier. It must not hard-filter candidates or
  count as binding, affinity, activity, pose, or validation evidence.
- Treat Vina as a pose generator and tool score. Never call its score experimental binding free
  energy, potency, or activity.
- Require PoseBusters validity before a pose can enter positive evidence. Use symmetry-aware RMSD
  only when an independent reference pose is legitimately available.
- Treat ProLIF/IFP agreement as method agreement, not proof of binding.
- Run OpenMM only for supported parameterizable, ordinary noncovalent systems. Record
  parameterization failure instead of imputing energy or stability.
- Treat the local 3D viewer, browser PNG, residues-within-5-angstrom list, sub-2-angstrom pair
  count, and docking-box containment as visual-QA aids. They are not PoseBusters, ProLIF,
  symmetry-aware RMSD, physical stability, affinity, or evidence of binding.

## Supported v1 scope

- One or two protein chains, total length at most about 700 residues.
- One ordinary noncovalent organic ligand, normally at most 100 heavy atoms.
- Explicitly reject or mark unsupported: covalent ligands, polymer ligands, metal-containing
  ligand chemistry, unsupported metal centers, and nonstandard systems that cannot be parameterized.
- Never silently rebuild long missing loops or replace uncertain stereochemistry.

## Evidence language

- `REDOCKING_RECOVERED`: a controlled redocking pose passed the declared reference-recovery gate.
- `METHOD_CONSENSUS`: independently generated valid methods agree under the declared IFP/pose rule.
- `HYPOTHESIS_ONLY`: geometrically plausible output lacks independent support.
- `REJECTED`: chemistry, geometry, clash, parameterization, provenance, or stage gate failed.

Every claim must cite an artifact ID or imported document location. When the tools did not measure
something, report it as unknown.
