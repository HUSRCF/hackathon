# Verification states

- `UNVERIFIED`: no identifier-bound identity comparison has completed.
- `EXACT_SEQUENCE`: one locally observed sequence exactly equals the accession-bound UniProt
  FASTA sequence.
- `CONSISTENT_VARIANT`: deterministic alignment is at least 98% identical with at least 90%
  reference coverage, but is not exact.
- `PARTIAL_COORDINATE_MATCH`: the observed coordinate sequence is an exact subsequence of the
  UniProt reference. This commonly reflects missing terminal or unresolved residues.
- `CONFLICT`: the supplied accession and best local observed sequence do not meet the above
  criteria.

The verification call sends only the user-supplied accession to `rest.uniprot.org`; alignment
is local. It never uploads a private sequence.

All states are limited to sequence identity. They do not establish structure accuracy,
biological assembly, protein state, ligand identity, binding, activity, pose correctness, or
docking readiness. A conflict blocks automatic use until the user resolves the identity.
