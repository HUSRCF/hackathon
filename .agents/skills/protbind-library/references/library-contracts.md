# Library contracts

## Layout

Protein and ligand roots are selected independently by the operator. Each root contains:

```text
catalog.sqlite
objects/
incoming/
quarantine/
derived/
receipts/
```

`objects/` is immutable SHA-256 content-addressed storage. `catalog.sqlite` stores bounded
metadata and artifact references. `receipts/` stores hash-bound scan and apply records.

## Import states

The implementation recognizes `DISCOVERED`, `STAGED`, `PARSED`, `QC_PASSED`, `QUARANTINED`,
`IDENTITY_CHECKED`, and `ACTIVE`. A normal successfully parsed import is ACTIVE but remains
identity-UNVERIFIED until a separate registry-bound check.

`QUARANTINED` means parse/QC could not establish a usable local record. It is not evidence that
the underlying biological claim is false. `workflow_v1_compatible=false` is a workflow support
boundary, not a parse failure.

## Plan/apply transaction

A plan commits to the source-relative name, byte size, SHA-256, format, target library root ID,
and skipped items. Applying rechecks size and hash before copying to the CAS. Move mode resolves
and re-hashes the stored artifact before deleting the source. It also requires the plan ID as a
second confirmation.

Never apply a plan if the user has not reviewed its counts or if the source changed. Copy is the
default. Catalog deduplication is by raw SHA-256 and does not overwrite an existing entry.

## CLI and Agent split

The CLI is the operator surface and accepts explicit paths:

```text
protbind library init
protbind library status
protbind library scan
protbind library import
protbind library list
protbind library show
protbind library verify-uniprot
```

Every CLI library operation requires `--confirm-data-access`. The Agent tools accept no
arbitrary path: they operate only on configured aliases and `incoming/`, and require
`data_access_confirmed=true` after a user prompt.
