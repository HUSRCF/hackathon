---
name: protbind-research
description: Operate ProtBind as an interactive, local-private, stage-gated protein-ligand research workflow. Use for creating or resuming ProtBind cases, fetching identifier-bound public candidates with exact-domain approval, inspecting screening/docking/validation progress, attaching approved support artifacts, handling failed scientific gates, and producing evidence-bounded reports without arbitrary shell, filesystem, or open network access.
---

# ProtBind Research

Control ProtBind through its typed MCP tools. Treat tool artifacts and gate receipts as the only
source of scientific state; never invent structures, scores, validation results, or completion.

## Run the closed loop

1. Read [stage-control.md](references/stage-control.md) before mutating a run.
2. Call `protbind_case_status` before every attempted advance.
3. Explain the returned stage, decision, failed checks, and required actions.
4. Call `protbind_case_advance` only with the fresh continuation token returned for `READY` or
   `RETRYABLE`. Execute one stage per call.
5. Require `acceptance.decision=ACCEPTED` before treating that stage as complete.
6. Inspect `next_gate` and repeat only when the user wants to continue. Never auto-retry a failure.
7. Stop on `NEEDS_ACTION`, `UNSUPPORTED`, or `FAILED`; resolve only the stated action, then request
   a fresh status. After `DOCKED`, use `protbind_case_pose_view` only for coordinate-free visual-QA
   metadata. At any checkpoint use `protbind_case_dossier` to distinguish computed stages from
   accepted stages. On `COMPLETE`, retrieve both the deterministic report and dossier.

Use `protbind_case_create` only for project-relative, offline case and index files. Use
`protbind_case_attach_support` only for a named, reviewed project-local artifact and respect stage
freeze rules. `protbind_fetch_public_data` is the only network-capable tool: use it only after the
user approves the exact source domain and supplies a public registry identifier plus a
project-relative destination. It never accepts an arbitrary URL or private sequence, and a fetched
file remains an unaccepted candidate until the normal case gates consume it. Read
[tool-contracts.md](references/tool-contracts.md) when fetching data, creating a case, attaching
support, or interpreting a tool error.

## Preserve scientific boundaries

Read [scientific-boundaries.md](references/scientific-boundaries.md) before interpreting docking,
cofolding, validation, or ranking evidence. State uncertainty and unsupported chemistry explicitly.
Do not turn Vina scores into experimental affinities or model poses into observed binding facts.
Never promote a screenshot, short-distance count, docking-box check, or manual visual impression
into a scientific gate; these only help an operator decide what artifact-backed check to inspect.

Do not use shell, generic file tools, subagents, or web/network tools as a workaround. Ask the user
to place reviewed inputs under the project root or use the identifier-only, exact-domain-approved
public fetch tool.

For local papers, ask before each project-document access. Call
`protbind_knowledge_document_inspect` first and disclose scan-like or unresolved OCR pages. Import
only after a separate confirmation with `protbind_knowledge_import`; use
`protbind_knowledge_search` for retrieval. Cite the returned artifact ID and one-based PDF page (or
Markdown section). Retrieval is not scientific validation, and a missing hit is not evidence of
absence. Never open generic shell/PDF/network tools through the Agent as a substitute.

When the user wants reusable protein/ligand storage, batch migration, catalog inspection, or
UniProt identity comparison, load `$protbind-library` and follow its separate per-call consent
gate. A library entry is not automatically an accepted case input: require ACTIVE state, resolve
any identity conflict or workflow compatibility blocker, attach the selected artifact through the
normal case contract, and then let the case stage gate revalidate it.
