---
name: protbind-research
description: Operate ProtBind as an interactive, local-private, stage-gated protein-ligand research workflow. Use for creating or resuming ProtBind cases, inspecting screening/docking/validation progress, attaching approved support artifacts, handling failed scientific gates, and producing evidence-bounded reports without arbitrary shell, filesystem, or network access.
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
freeze rules. Read [tool-contracts.md](references/tool-contracts.md) when creating a case,
attaching support, or interpreting a tool error.

## Preserve scientific boundaries

Read [scientific-boundaries.md](references/scientific-boundaries.md) before interpreting docking,
cofolding, validation, or ranking evidence. State uncertainty and unsupported chemistry explicitly.
Do not turn Vina scores into experimental affinities or model poses into observed binding facts.
Never promote a screenshot, short-distance count, docking-box check, or manual visual impression
into a scientific gate; these only help an operator decide what artifact-backed check to inspect.

Do not use shell, generic file tools, subagents, or web/network tools as a workaround. Ask the user
to place reviewed inputs under the project root or to perform a separately authorized import.
