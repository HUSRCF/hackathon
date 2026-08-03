# Optional MMseqs2 protein homology support

ProtBind now exposes MMseqs2 as an optional, local-only protein sequence utility:

- `homology search` searches a private query FASTA against a private target FASTA;
- `homology cluster` creates a local sequence-cluster assignment TSV;
- both commands emit a hash-bound receipt containing parameters, input/output hashes and row counts.

The command is not part of the docking state machine. It never changes a case manifest, box,
screening threshold, pose, or scientific conclusion. Its intended uses are:

1. retrieving homologous proteins and possible structure/cache reuse;
2. removing redundant protein entries from a local library;
3. producing an upstream sequence-cluster assignment for a future leakage-resistant study.

Example:

```bash
protbind homology search \
  --query private/query.fasta \
  --target private/proteins.fasta \
  --output artifacts/homology/hits.tsv \
  --receipt artifacts/homology/search-receipt.json \
  --min-seq-id 0.30 \
  --coverage 0.80 \
  --cov-mode 0 \
  --threads 8 \
  --confirm-data-access
```

For a future cluster-backed split protocol:

```bash
protbind homology cluster \
  --input private/proteins.fasta \
  --assignments private/protein-clusters.tsv \
  --receipt artifacts/homology/cluster-receipt.json \
  --min-seq-id 0.30 \
  --coverage 0.80 \
  --cov-mode 0 \
  --threads 8 \
  --confirm-data-access
```

`--min-seq-id 0.30` is an MMseqs2 parameter only when the actual MMseqs2 executable runs. It
must not be confused with ProtBind's independent `global_edit_identity` audit. The receipt records
the exact MMseqs2 version and parameters but does not expose raw sequences or sequence identifiers.
MMseqs2 is not an affinity predictor, docking engine, or AMD GPU performance benchmark.
