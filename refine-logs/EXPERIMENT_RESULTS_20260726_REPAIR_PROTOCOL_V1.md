# ProtBind `repair-protocol-v1` frozen ten-case rerun — 2026-07-26

## Decision and scope

The conservative receptor repair rule was promoted from the two-case retrospective ablation to an
explicit protocol revision and rerun over the complete, unchanged ten-case holdout. No case was
substituted and failed cases remain in the denominator. This revision is frozen as
`repair-protocol-v1`; repair-enabled batch runs now require an explicit lower-case revision, and a
different revision cannot resume its terminal artifacts.

This is a controlled revision on a previously observed holdout, not a new prospective holdout. It
tests whether the declared repair rule behaves consistently across all ten cases; it does not
provide an unbiased estimate of tuning benefit. The original 2026-07-25 7/10 baseline remains
immutable and separately reported.

## Frozen protocol

- Holdout file SHA-256: `01d9fd57f31ef006601b6a1e982d2cf020d50761bc9c8f5bfe61497ccc064ca3`.
- Holdout selection SHA-256: `242f43398baa76fee2b6b7ab0e53546cd066597cb3f638f81be4378dc8146a89`.
- ProtBind 35-file source manifest SHA-256:
  `0fcb6c2bad48da233b0b4a543cd88e32b6eac5d5fff7c49d8c93b02e0f223741`.
- Internal run-plan SHA-256:
  `bb714243c4ec860694dde0d10f0182b7c48a192d6fcdac42be0672fd787b903f`.
- Repair: PDBFixer standard-residue heavy atoms only, native-ligand 6 Å protected region, no loop
  reconstruction, no hydrogen addition, no `--allow_bad_res`, and original-heavy-atom coordinate
  delta required to remain zero.
- Vina: seed `20260721`, 5 Å box padding, exhaustiveness 32, 9 modes, energy range 3 kcal/mol,
  one CPU per case, 1800 s timeout, at most two concurrent cases.
- Vina SHA-256: `f31f774f723bba7bbe6e9d1c47577020eea9a8da16424284c043d22593570644`.
- Meeko receptor/ligand/export SHA-256:
  `c8bed9bd996ad6e665e525e9e077c6866eca6c06d8a2a393db34069dec3d77b6`,
  `6d9ef4a4e51079aa93e6c49c4476321e82038e45fdd5e8ed4210f2d37b47f578`, and
  `fce9eff3bebaf46af04660a2dace62113595682dbbfd9a8a1bbff8734b6e1919`.

## Result

All ten cases reached a terminal state: 9 completed and 1 failed closed. Independent real-tool
PoseBusters 0.6.5, sPyRMSD 0.9.0 and ProLIF 2.2.0 evaluation had zero metric failures on the nine
completed cases. With the frozen denominator of ten, PB-valid plus symmetry-aware RMSD ≤2 Å was
8/10 at top 1 and 8/10 at the top-5 oracle. The gate remains incomplete because 7YZU never reached
docking.

| Case | Terminal result | Independent top-1 RMSD (Å) | Top-1/top-5 recovered | IFP Jaccard |
|---|---|---:|---|---:|
| 7XFA_D9J | completed | 9.2686 | no / no | 0.0909 |
| 7AN5_RDH | completed | 0.5345 | yes / yes | 0.6000 |
| 7WQQ_5Z6 | completed | 0.4618 | yes / yes | 0.8333 |
| 7BTT_F8R | completed after repair | 0.9270 | yes / yes | 0.8000 |
| 7YZU_DO7 | receptor preparation failed | — | no / no | — |
| 8BOM_QU6 | completed | 1.2914 | yes / yes | 0.5714 |
| 6YQW_82I | completed | 0.6001 | yes / yes | 1.0000 |
| 7THI_PGA | completed | 0.7223 | yes / yes | 0.7778 |
| 7DUA_HJ0 | completed | 1.1377 | yes / yes | 0.6250 |
| 7ELT_TYM | completed | 0.3960 | yes / yes | 0.5000 |

Across the nine evaluated cases, IFP Jaccard mean/median were 0.6443/0.6250. Protein-ligand
pairwise clash count was zero for all nine. These are structural diagnostics, not affinity
evidence.

## Repair receipts and unresolved failure

Only 7BTT and 7YZU required heavy-atom repair. 7BTT repaired 13 heavy atoms in three residues
outside the protected pocket and preserved every original heavy-atom coordinate; it then recovered
the native pose at 0.9270 Å. The other eight docked receptors added no heavy atoms and also had zero
original-heavy-atom displacement.

7YZU repaired 63 heavy atoms in 14 outside-pocket residues, removed pre-existing receptor
hydrogens so Meeko remained the sole protonation authority, and preserved original heavy-atom
coordinates. Meeko receptor preparation still exited nonzero. ProtBind therefore retained
`TOOL_NONZERO_EXIT` at `receptor_preparation`; it did not delete residues, use `--allow_bad_res`, or
manufacture a docking result.

## Immutable artifacts

| Artifact | File SHA-256 | Internal content SHA-256 |
|---|---|---|
| [`run-plan.json`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/run-plan.json) | `a488542db18d127e274f517b275557b34104046097419ba8519bb48ab06d2a7b` | `bb714243c4ec860694dde0d10f0182b7c48a192d6fcdac42be0672fd787b903f` |
| [`batch-result.json`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/batch-result.json) | `248a24f8d9ecabfdc7e2a5564853d5f830aa1c6a722cf8980a258d77133fb28a` | `7b014928c0688f1a64a661b153e0c846ee2a50ef181eab8900dd48714d07a5bf` |
| [`regression-manifest.json`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/regression-manifest.json) | `b6c5f3df6422090d3e485e711537dc05c6a165e0e88011fe5f6693e6445a463c` | `f289a0b35c0184363af16cbd5bcfd69dc0e5b93209f38659d9dd17aee5ade91f` |
| [`independent-regression.json`](../experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/independent-regression.json) | `e9c77eaf01583f421583971b626b1dcdba7a2e3ca94983bce3c3f935e7f4391c` | `076643a00d761dc00afc9a81e64ca69c509eba28743e49c8ad0b786e51459abc` |

## Claim boundary and next gate

This is known-site redocking, not blind docking, virtual-screening enrichment, hit rate, or binding
affinity. Top-5 is an oracle recovery diagnostic. Vina scores are not experimental free energies;
PoseBusters energy fields are conformational diagnostics; IFP overlap is not binding evidence.

The next scientific decision for 7YZU must be a separately named and frozen protocol (for example,
a restrained side-chain geometry procedure with explicit cofactor/heterogen policy), validated
before use and rerun without case substitution. Until then, `repair-protocol-v1` remains a 9/10
completed, 8/10 recovered, incomplete-gate result.
