---
name: protbind-drutai
description: Operate ProtBind's optional DrutAI sequence-SMILES DTI concordance annotator. Use when inspecting DrutAI availability, acquiring a fixed ONNX model with explicit GPL acknowledgement, running a private local annotation, or interpreting SUPPORTIVE, DISCORDANT, ABSTAIN, and FAILED outcomes without treating them as binding evidence.
---

# ProtBind DrutAI

Keep DrutAI optional, local, auditable, and non-decisional. Never copy upstream implementation
code into ProtBind, never distribute model weights with ProtBind, and never use a score to hard
filter a candidate.

## Inspect before acting

1. Call `protbind_drutai_status` before acquisition or annotation.
2. Report whether the requested model is present and hash-valid.
3. State that DrutAI uses sequence and SMILES descriptors rather than a pocket or pose.
4. Preserve its `annotation-only` scientific role.

## Acquire one model

Call `protbind_drutai_model_acquire` only after showing the exact model, fixed source commit,
`raw.githubusercontent.com` network target, private cache write, and conservative GPL-3.0-only
handling. Require the exact `GPL-3.0-only` acknowledgement. Never substitute another URL, commit,
license string, or unpinned weight.

Treat the acquisition receipt as provenance only. It does not admit the model as binding evidence.

## Run a private annotation

1. Require a project-relative TSV with `sm`, `target`, and `smile` columns plus one
   `{target}.fasta` file per target.
2. Obtain fresh private-data approval before reading the TSV or FASTA files.
3. Call `protbind_drutai_annotate`; do not invoke shell, arbitrary executables, or network tools.
4. Stop on invalid SMILES, noncanonical/ambiguous FASTA, missing model, missing network isolation,
   worker failure, output identity drift, or malformed probabilities.
5. Cite the returned bundle artifact when reporting results.

## Interpret outcomes

- `SUPPORTIVE`: model direction agrees with the structural-screening candidate hypothesis.
- `DISCORDANT`: model direction conflicts; keep the candidate and request review.
- `ABSTAIN`: the declared uncertainty band prevents a directional annotation.
- `FAILED`: control or execution failed; no scientific result exists.

Never promote any outcome to direct binding, affinity, activity, pose validity, or experimental
evidence. Never upgrade an evidence grade from DrutAI alone. Read
[scientific-boundaries.md](references/scientific-boundaries.md) before drafting a claim.

## Preserve data and license boundaries

- Keep raw sequences, SMILES, and worker output inside the private content-addressed workspace.
- Return commitments and artifact references rather than sequence text.
- Run the separate executable with its cache disabled and OS-level network isolation enabled.
  Ordinary binaries require bubblewrap `--unshare-net`. A canonical `/snap/bin` worker may use
  Snap confinement only after every call verifies strict confinement, non-devmode/non-trymode,
  and no connected `network*` interface; otherwise fail closed.
- Do not delete or replace an acquired model without a fresh approval and exact target.
- Treat publication/original-output parity as already verified by the project maintainer; do not
  rerun it unless the pinned source commit, model bytes, or feature contract changes.
