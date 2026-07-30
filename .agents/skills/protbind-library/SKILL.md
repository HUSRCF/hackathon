---
name: protbind-library
description: Safely manage ProtBind private protein and ligand libraries. Use when a user wants to choose reusable data locations, inspect incoming PDB/mmCIF/FASTA/SDF/SMILES files, plan or apply copy/move imports, review QC or quarantine results, or verify a protein sequence against an explicitly supplied UniProt accession.
---

# ProtBind Library

Operate the content-addressed protein and ligand catalogs through the bounded ProtBind MCP
tools. Treat local files as untrusted candidates until parsing, QC, and any requested identity
check have completed.

## Consent gate

Before every library tool call:

1. State which library alias and operation will be accessed.
2. State whether the operation reads metadata, hashes file contents, copies data, deletes a
   verified source after copying, or contacts `rest.uniprot.org`.
3. Ask the user for a fresh confirmation. After they agree, pass
   `data_access_confirmed=true`.

Do not reuse a confirmation for a later tool call. OpenCode may also show its own `ask` prompt;
both controls are intentional.

## Closed-loop import

1. Call `protbind_library_status`. If it is unconfigured, stop and give the operator the exact
   `protbind library init` and MCP `--library-config` steps. Never choose library roots yourself.
2. Ask the user to place files in the relevant preconfigured `incoming/` directory. The Agent
   cannot browse an arbitrary source path.
3. Call `protbind_library_plan_import`. Present the plan ID, included count, skipped items, and
   the fact that nothing has been imported yet.
4. Ask separately before calling `protbind_library_apply_import`.
5. Default to `mode=copy`. Use `mode=move` only when the user explicitly asks to migrate/remove
   sources, and pass the exact plan ID as `confirm_move`.
6. Inspect every result. Report ACTIVE, QUARANTINED, deduplicated, and workflow compatibility
   separately. Stop on quarantine or conflict; do not advance a research case with that entry.

Read [library-contracts.md](references/library-contracts.md) before planning or applying an
import. Read [verification-states.md](references/verification-states.md) before describing QC
or UniProt results.

## Identity verification

Ask for a specific UniProt accession. Then explain that only the accession is sent to
`rest.uniprot.org`; the private local sequence remains local. Call
`protbind_library_verify_uniprot` only after the separate data/network permission prompt.

Never perform remote similarity search or sequence upload. `EXACT_SEQUENCE` authenticates only
the compared observed sequence. It does not authenticate the coordinate model, assembly,
ligand, biological activity, or docking suitability.

## Library RAG

`protbind_library_rag_sync` and `protbind_library_rag_search` are private-library reads, so the
fresh consent gate applies to every call. Sync creates a derived seekdb projection containing
entry IDs, catalog/QC states, accession when already verified, bounded counts, and workflow
blockers. It excludes filenames, paths, sequences, SMILES, molecule bytes, and coordinates.

Treat search results only as candidate discovery. Cite both the projection artifact ID and entry
ID, then call `protbind_library_show` with a new confirmation and re-run normal case gates before
using an entry. `catalog.sqlite` remains the exact state source; the vector/full-text index may be
stale and never overrides it. Do not answer binding, activity, identity, or structure-quality
questions from embedding similarity.

## Scientific and security boundaries

- Do not call parsed files “true”, “validated”, “binding”, or “active”.
- Do not treat P2Rank sites as known binding sites or DrutAI output as binding evidence.
- Never expose absolute roots, private sequence text, molecule bytes, or coordinates in chat.
- Never overwrite source files or catalog objects.
- Never use arbitrary shell/file/network tools as a substitute for a ProtBind tool.
- Never request broad root, `sudo`, setuid, `chmod 777`, or passwordless privilege. If an
  operator-owned root cannot be created, stop and request a narrowly scoped installation or
  filesystem action outside the Agent.
