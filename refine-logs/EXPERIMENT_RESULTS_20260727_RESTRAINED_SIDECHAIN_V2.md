# Restrained side-chain repair protocol v2 — 2026-07-27

## Outcome

`repair-protocol-v2-restrained-sidechain` was frozen and run on the same immutable
ten-complex holdout used by the original and v1 protocols. All ten cases reached a
terminal docking result, all ten completed independent PoseBusters/sPyRMSD/ProLIF
evaluation, and the mechanical regression gate is complete:

- 10/10 attempted, 10/10 completed, 0 preparation or metric failures;
- PB-valid and symmetry-RMSD ≤2 Å: top-1 9/10 and top-5 oracle 9/10;
- IFP Jaccard mean/median 0.6598/0.7014;
- zero independently measured protein–ligand pairwise clashes in all ten top-1 poses;
- `gate_complete=true`, with no gate blockers.

This is a controlled protocol revision on an already observed holdout. It is not a
new prospective estimate, a blind-pocket benchmark, a virtual-screening hit rate, an
affinity result, or evidence that any ligand binds experimentally. The immutable
original result remains 7/10, and v1 remains 8/10.

## Why v2 was needed

The v1 conservative repair restored missing standard-residue heavy atoms only when
all missing atoms were outside the native-ligand 6 Å protected region. It recovered
7BTT but 7YZU still failed in Meeko/RDKit with an explicit-valence error after 63
outside-pocket side-chain atoms were added. This was a repair-geometry problem, not a
categorical declaration that the receptor class was unsupported.

v2 adds a constrained geometry layer between conservative PDBFixer repair and Meeko:

1. identify original and newly added heavy atoms by residue/atom identity;
2. add transient hydrogens for force-field preparation;
3. assign zero mass to every original heavy atom, so only newly added heavy atoms and
   transient hydrogens can move;
4. minimize with OpenMM `amber14-all.xml`, `CutoffNonPeriodic` 10 Å, CPU/1 thread;
5. remove all hydrogens before passing the receptor to Meeko;
6. fail closed unless original-heavy identity and coordinates, added-atom bond
   geometry, nonbonded distance, and heavy-atom chirality checks all pass.

The default iteration schedule is `250 → 1000 → 5000`. A later attempt is allowed
only when constrained geometry has not converged or Meeko/RDKit reports a narrow
valence/sanitization chemistry failure. Tool crashes, identity changes, unsupported
force-field templates, and other failures are not retried as if they were
convergence failures. The first accepted attempt becomes the exact Meeko input and
all attempts are recorded.

OpenMM energies in these receipts are preparation-only force-field diagnostics. They
are not binding energies, free energies, or physical stability claims.

## Frozen configuration

- Protocol revision: `repair-protocol-v2-restrained-sidechain`
- Holdout file SHA-256:
  `01d9fd57f31ef006601b6a1e982d2cf020d50761bc9c8f5bfe61497ccc064ca3`
- Holdout selection SHA-256:
  `242f43398baa76fee2b6b7ab0e53546cd066597cb3f638f81be4378dc8146a89`
- Conservative repair: enabled, native-ligand protected radius 6 Å
- Restrained side-chain optimization: enabled
- Iteration limits: 250, 1000, 5000
- Vina: seed 20260721, box padding 5 Å, exhaustiveness 32, 9 modes,
  energy range 3 kcal/mol, CPU 1, timeout 1800 s
- Maximum parallel cases: 2
- Frozen source-manifest SHA-256:
  `56cc89f42e8397340b64546bc1824adbe80730eab0b5ae1a632fbf2d1c718e5d`

The run used the existing AIAA environment. This preparation and Vina regression is
CPU-only; it did not reserve or consume either Radeon GPU. The available OpenMM build
exposed CPU/Reference platforms, not HIP.

## Independent per-case result

| Case | Completed | PB-valid | Top-1 RMSD Å | Top-1 recovered | Top-5 recovered | IFP Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| 7XFA_D9J | yes | yes | 9.269 | no | no | 0.091 |
| 7AN5_RDH | yes | yes | 0.535 | yes | yes | 0.600 |
| 7WQQ_5Z6 | yes | yes | 0.462 | yes | yes | 0.833 |
| 7BTT_F8R | yes | yes | 0.927 | yes | yes | 0.800 |
| 7YZU_DO7 | yes | yes | 0.791 | yes | yes | 0.800 |
| 8BOM_QU6 | yes | yes | 1.291 | yes | yes | 0.571 |
| 6YQW_82I | yes | yes | 0.600 | yes | yes | 1.000 |
| 7THI_PGA | yes | yes | 0.722 | yes | yes | 0.778 |
| 7DUA_HJ0 | yes | yes | 1.138 | yes | yes | 0.625 |
| 7ELT_TYM | yes | yes | 0.396 | yes | yes | 0.500 |

The top-1 symmetry-RMSD mean/median are 1.613/0.757 Å. The sole pose-recovery failure
is 7XFA; its top-1 pose is PB-valid but is 9.269 Å from the reference.

## Constrained-repair receipts

Only 7BTT and 7YZU required the new layer. Both were accepted at the first 250
iteration limit and then passed the real Meeko/RDKit preparation.

| Check | 7BTT_F8R | 7YZU_DO7 |
|---|---:|---:|
| Fixed original heavy atoms | 2385 | 2915 |
| Mobile newly added heavy atoms | 13 | 63 |
| Transient hydrogens | 2377 | 2972 |
| Hydrogens passed to Meeko | 0 | 0 |
| Max original-heavy displacement Å | 0.000 | 0.000 |
| Chirality centers/signs preserved | 314/314 | 394/394 |
| Minimum nonbonded vdW ratio | 0.714 | 0.706 |
| Required minimum ratio | 0.600 | 0.600 |
| Accepted iteration limit | 250 | 250 |

7BTT optimization-receipt SHA-256:
`6dac2d89b7002a143379c0571bae03c2c4fade815c7234574010dd0088e8cc32`.
7YZU optimization-receipt SHA-256:
`0a656949d96cc0e73713cbbc9f03230db6818465974010a75f8761d39538c9b0`.

## Frozen artifacts

| Artifact | File SHA-256 | Internal SHA-256 |
|---|---|---|
| `run-plan.json` | `c6d6b4117b8db4503ad535530195a70b0f439847c6e6bba86a3c0b21634c813c` | `26bf03ed973b18bfde4c3f5af4b58da5d5d2e5131b83f20795df95375af18e68` |
| `batch-result.json` | `43e413f46bea1396ee653f250c0e789b1022821354807ff14ecf91d14bec75a1` | `566128395b794b488cfbc5153b0b599a0b5676eb3d532d6d4eee3918a39a0692` |
| `regression-manifest.json` | `09d5b714f7cb5658d186b3bb2d0db6f136708125c697405a01db0b9a66adb818` | `45a08672014af2f359bbd1c3484d234ed9bc931779351f087f9127c039b930f3` |
| `independent-regression.json` | `90e558fc93657b15dd5bf46862aa19e6d20e36cccef62d1520aa38ab6c9c8de1` | `5f524930248aebe717a316a0a44c89be95e805b354ef5b00eaaf7436d3c70b35` |

The independent evaluator re-read the frozen results and recomputed PoseBusters,
symmetry-aware RMSD, local-crop ProLIF IFP, clash, and strain diagnostics with the
real tools; it did not trust the batch summary as the metric source.

## Verification

- Ruff: pass
- Python compileall for `src`, `workers`, and `tests`: pass
- Full pytest suite: 273/273 pass

