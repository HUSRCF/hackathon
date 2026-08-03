# DrutAI scientific boundaries

DrutAI consumes a protein sequence and a small-molecule SMILES string. It does not inspect a
three-dimensional receptor, binding pocket, ligand conformer, or pose. Its output is therefore an
orthogonal sequence-SMILES model annotation, not a physical verification of binding.

Use the score only to describe method concordance. Do not:

- convert it into affinity, potency, activity, selectivity, or target-engagement evidence;
- discard a candidate solely because it is `DISCORDANT`;
- count related DrutAI architectures as independent biological evidence;
- treat the fixed 0.5 model class threshold as calibrated probability;
- infer absence of binding from a negative prediction.

The model remains pending external applicability-domain, leakage-controlled, calibration, and
public positive/inactive bake-off. Until those gates pass, all annotations have
`decision_eligible=false` and `hard_filter_allowed=false`.
