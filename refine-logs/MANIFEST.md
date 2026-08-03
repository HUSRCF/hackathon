# Experiment artifact manifest — updated 2026-08-03

| Relative artifact | SHA-256 | Interpretation |
|---|---|---|
| `experiment-results/aiaa-environment.json` | `3c0500a6c7f303bccfbdff63cdae3d5f98c49c14fcc39ecf0c1c99e7ba6c3200` | Two-GPU AIAA/core audit |
| `experiment-results/aiaa-openfold3-environment.json` | `98f71a60f57351685f3dfc660f31e7d148042c40ccacfe645faa586ce9651151` | Single-selected-GPU OpenFold runtime audit |
| `experiment-results/aiaa-selection-quick-vina-20260723/environment.json` | `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35` | Historical v1 path-redacted AIAA/core lock; quick Vina was CPU-only |
| `experiment-results/aiaa-selection-quick-vina-20260723/vina-provenance.json` | `7edb484b4f201a7b3826b99fc16ba5e1711887c464102e55931214dbe2e7c30d` | Historical v1 full/quick Vina identities and frozen CPU profile |
| `experiment-results/aiaa-selection-quick-vina-20260723/smoke-result.json` | `1280f3f06a3504a6bac626f85eb1c8ba8ef42eba0252cad61cd13544db315c00` | Historical v1 one-request direct-adapter smoke; no scientific claim |
| `experiment-results/aiaa-selection-quick-vina-20260723-v2/environment.json` | `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35` | Historical v2 path-redacted AIAA/core lock; quick Vina was CPU-only |
| `experiment-results/aiaa-selection-quick-vina-20260723-v2/vina-provenance.json` | `d1b17d7c43d034ee33310699780b79b47a03555a9b8f2a6dc979567342f7a0a5` | Historical v2 `selection-quick-vina-1.1` and full-Vina code identities |
| `experiment-results/aiaa-selection-quick-vina-20260723-v2/smoke-result.json` | `962d6fe42b4a4c66f426d2d7c754817c2e51cf9084a07e7408886f242c25ee7a` | Historical v2 3/3-request direct-adapter protocol smoke; no scientific claim |
| `experiment-results/aiaa-selection-quick-vina-20260723-v2/verified-smoke-index.sqlite` | `dda189cb5db528e7422a027340fbe643e4bae329493220fd789e50d7c628f8a7` | Historical v2 chemistry-verified production-isolation smoke index |
| `experiment-results/aiaa-selection-quick-vina-20260723-v2/production-workspace/runs/production-isolation-smoke-v2c/manifest.json` | `568449c52f98d0a0603704e528844292aa2475f2e6287fd68b0b4f771cd50c4a` | Historical v2c host-blocked production-isolation manifest |
| `experiment-results/aiaa-selection-quick-vina-20260723-v3/environment.json` | `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35` | Historical v3 path-redacted AIAA/core lock; two W7900 detected; quick Vina was CPU-only |
| `experiment-results/aiaa-selection-quick-vina-20260723-v3/vina-provenance.json` | `b9b226eb718c2435f7450395f1ac40c2b1ae27a42ce40b02b8a316edfbae1536` | Historical `selection-quick-vina-1.2` and full-Vina code identities |
| `experiment-results/aiaa-selection-quick-vina-20260723-v3/smoke-result.json` | `c0e077c2d8c24e59fc4f6d3eece777f1c455b5fd325a7c890152d724339c11ee` | Historical direct 3/3-request smoke; superseded by selection-2.5 v4 |
| `experiment-results/aiaa-selection-quick-vina-20260723-v3/direct-workspace/objects/65/3249f4f516e4cde020ce82f38a963fa75cccf4800282f03e98d2cd72c18757` | `653249f4f516e4cde020ce82f38a963fa75cccf4800282f03e98d2cd72c18757` | Historical v3 direct box receipt; receptor-frame geometry sanity only, no overlap/site validation |
| `experiment-results/aiaa-selection-quick-vina-20260723-v3/verified-smoke-index.sqlite` | `dda189cb5db528e7422a027340fbe643e4bae329493220fd789e50d7c628f8a7` | Tiny `chemistry_verified=true` isolation fixture used by the latest production-isolation attempt; not a scientific screening library |
| `experiment-results/aiaa-selection-quick-vina-20260723-v3/production-workspace/objects/64/1d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a` | `641d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a` | Latest production-isolation attempt's box receipt, bound by preparation/input; sanity only |
| `experiment-results/aiaa-selection-quick-vina-20260723-v3/production-workspace/runs/production-isolation-smoke-v3/manifest.json` | `730e54551f807ed18896257fa3d2f47ba9da3a930c8bc2b7b8a2c6ebff585746` | Latest production-isolation attempt reached `SELECTED`, then host bwrap loopback failed closed: `DEGRADED`, last `SCREENED`, no worker/science result |
| `experiment-results/aiaa-selection-quick-vina-20260725-v4/vina-provenance.json` | `b4f51a5b93e5ac676edb328d7f86b177c009c7e90a11b8a9e444d0f0e0c7739f` | Current quick-1.3/full-Vina provenance; selection host contract is 2.5 |
| `experiment-results/aiaa-selection-quick-vina-20260725-v4/smoke-result.json` | `8b19a1d9aa76af0f60dcf0d95c17aa86357598b315d40478dd59c238da0e6f8e` | Current direct 3/3-request, 36-entry closure; user-center/unverified-chemistry/application-offline only |
| `configs/redock-regression-pilot-20260723.json` | `9bcaa75605ff2687ce332c6206b449093283ad939f09021dd33ce3d10c0d5b8e` | Six-case retrospective pilot manifest; not a frozen ten-case holdout |
| `experiment-results/redock-regression-pilot-20260725/result.json` | `241c1dc039802b1445d595d00fa771269dff70602c61a17f70238b7552e0f27b` | Regression 1.1: 6 attempted, 3 completed, 3 fail closed; top-1 2/6, top-5 3/6, IFP/strain diagnostics |
| `experiment-results/posebusters-redock-holdout-20260725/holdout.json` | `01d9fd57f31ef006601b6a1e982d2cf020d50761bc9c8f5bfe61497ccc064ca3` | Result-blind fixed-ten selection from 308 IDs; 298 explicit exclusions, no result access or case substitution |
| `experiment-results/posebusters-redock-fixed10-formal-20260725/run-plan.json` | `a3eb1736d63fb76086c111fddb56fff75793d307c76cc4f409a2e7fe266ff99c` | Frozen pre-run holdout/input/code/tool/config bindings; internal plan `fb41f3fd...` |
| `experiment-results/posebusters-redock-fixed10-formal-20260725/batch-result.json` | `89deee7806792f120ef23b6cf72aab8bd910c54ac6b929fbfa5b2dbd1a36ad06` | Formal 10/10 attempted batch: 8 completed, 2 fail closed, top-1/top-5 7/10 |
| `experiment-results/posebusters-redock-fixed10-formal-20260725/regression-manifest.json` | `34d72fa21156ade868486dda6df791cc4bc2ccc6a44351417882039bb6af8b56` | Exact ten-case independent-evaluation manifest; internal manifest `86f213ae...` |
| `experiment-results/posebusters-redock-fixed10-formal-20260725/regression-v2.json` | `cab37219c7918a852a35b0296199da8cc60e5bdb52c584f72a8d517e232a85a8` | Authoritative fixed-ten PB/RMSD/strain and local-crop IFP result; formal 7/10, gate incomplete |
| `experiment-results/posebusters-redock-repair-remediation-20260725/7btt_f8r/result.json` | `f1efbe057f47a40075d9b18883815efaa73568df217fdf4cda2b51948815bce7` | Retrospective repair only: 13 outside-pocket heavy atoms added; top-1 recovered at 0.9270 Å |
| `experiment-results/posebusters-redock-repair-remediation-20260725/7yzu_do7/result.json` | `f84b19fa767e075594b26652fff445e23df8604633c2553f70e149ae129660b0` | Retrospective repair remains fail closed; no silent bad-residue deletion |
| `experiment-results/posebusters-redock-repair-remediation-20260725/regression-manifest.json` | `372b60c210657d29bc1ab23fd65a1490fb7f70ba0193d23961904954dcd8fb95` | Two-case retrospective remediation manifest; not a frozen holdout |
| `experiment-results/posebusters-redock-repair-remediation-20260725/regression-v2.json` | `5fba720c25196b3584f4b03c32622f9beb11bcc30315ce7752e9d66f3c5de7aa` | Local-crop remediation evaluation: 1/2; must not update the formal fixed-ten score |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/run-plan.json` | `a488542db18d127e274f517b275557b34104046097419ba8519bb48ab06d2a7b` | Frozen all-ten `repair-protocol-v1`; internal plan `bb714243...`, same holdout, no substitution |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/batch-result.json` | `248a24f8d9ecabfdc7e2a5564853d5f830aa1c6a722cf8980a258d77133fb28a` | Revised protocol 10/10 terminal: 9 completed, 1 fail closed, recorded top-1/top-5 8/10; internal result `7b014928...` |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/regression-manifest.json` | `b6c5f3df6422090d3e485e711537dc05c6a165e0e88011fe5f6693e6445a463c` | Exact revised ten-case independent-evaluation manifest; internal manifest `f289a0b3...` |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/independent-regression.json` | `e9c77eaf01583f421583971b626b1dcdba7a2e3ca94983bce3c3f935e7f4391c` | Real PB/sPyRMSD/ProLIF recomputation: top-1/top-5 8/10, 0 metric failures, gate incomplete; internal result `076643a0...` |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/run-plan.json` | `c6d6b4117b8db4503ad535530195a70b0f439847c6e6bba86a3c0b21634c813c` | Frozen all-ten restrained-side-chain v2; internal plan `26bf03ed...`, iterations 250/1000/5000, same holdout and no substitution |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/batch-result.json` | `43e413f46bea1396ee653f250c0e789b1022821354807ff14ecf91d14bec75a1` | v2 10/10 terminal and completed, recorded top-1/top-5 9/10; internal result `56612839...` |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/regression-manifest.json` | `09d5b714f7cb5658d186b3bb2d0db6f136708125c697405a01db0b9a66adb818` | Exact v2 ten-case independent-evaluation manifest; internal manifest `45a08672...` |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/independent-regression.json` | `90e558fc93657b15dd5bf46862aa19e6d20e36cccef62d1520aa38ab6c9c8de1` | Real PB/sPyRMSD/ProLIF recomputation: top-1/top-5 9/10, 0 failures, `gate_complete=true`; internal result `5f524930...` |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/cases/7btt_f8r/store/objects/6d/ac2d89b7002a143379c0571bae03c2c4fade815c7234574010dd0088e8cc32` | `6dac2d89b7002a143379c0571bae03c2c4fade815c7234574010dd0088e8cc32` | v2 constrained-geometry receipt: 2385 original heavy atoms fixed, 13 added heavy atoms mobile, chirality/distance gates pass |
| `experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/cases/7yzu_do7/store/objects/0a/656949d96cc0e73713cbbc9f03230db6818465974010a75f8761d39538c9b0` | `0a656949d96cc0e73713cbbc9f03230db6818465974010a75f8761d39538c9b0` | v2 constrained-geometry receipt: 2915 original heavy atoms fixed, 63 added heavy atoms mobile, chirality/distance gates pass |
| `experiment-results/known-site-calibration-1iep-20260725/objects/65/0aff4a6910ec549f1b807f80d8c3ef424be9461818c5b5886cd6bfbffa4d82` | `650aff4a6910ec549f1b807f80d8c3ef424be9461818c5b5886cd6bfbffa4d82` | Real 1IEP target-specific known-site calibration PASS receipt |
| `experiment-results/known-site-calibration-1iep-20260725/objects/48/9c594be042ff257046ada66dffbd69fb9659ace14589308a9a916f1a281d17` | `489c594be042ff257046ada66dffbd69fb9659ace14589308a9a916f1a281d17` | Selection-2.5 preparation that consumed the exact 1IEP calibration |
| `experiment-results/known-site-calibration-1iep-20260725/objects/26/836f7ca9832907924469d0fdc66d6edc753f2d55b1c31a0bd86c2fe15c1d1b` | `26836f7ca9832907924469d0fdc66d6edc753f2d55b1c31a0bd86c2fe15c1d1b` | Minimal quick input bound to calibrated preparation; no native reference |
| `experiment-results/esmfold-aiaa-attested-receipt.json` | `4c26e4049880c81a9bae85c480d705f178059ce7d0b5f4f2600eaee0b07b1445` | Path-free 24-aa offline ESMFold v1 AIAA smoke receipt |
| `experiment-results/tripharm-hip-gfx1100-100k.json` | `89afcba17cd5d8e16b9fcbf4008f5de84be1d22ebb15c72f0c57e9f2b96fda38` | Triangle microbenchmark only |
| `experiment-results/demo-index.sqlite` | `6470368f70b3192836a746da03211b384794a762f9af4cd85657120f685f7919` | Synthetic unverified-chemistry fixture index |
| `experiment-results/vina-smoke/runtime-attestation.json` | `ce409fc219f115598e1fd9ace4a2ac045853c4bff1c84c1b2ba6bddd08ed009a` | Local Vina/Meeko scientific package attestation |
| `experiment-results/vina-smoke/ethanol-vina.sdf` | `037b3642daea45b9c3aef22a7f1c8598b43476781e7c6c3144f620c21cd840ca` | Toy Vina modes; no binding claim |
| `experiment-results/vina-smoke/posebusters.csv` | `ed4e4d84521f9f5586d1e1a38981d276fec525312994df11efde8b3ac9ef84ab` | Toy-pose parsing/gate output |
| `experiment-results/redock-1iep-20260723-final/result.json` | `9f767a8baa5771bf5eb1a7dce03382661668d87ea81fef71362e4b2c9d32cc67` | Public known-site redocking; top-1 recovered |
| `experiment-results/redock-1s19-mc9-20260723-final/result.json` | `af2b5b81e952bd9205760f7f15f9468cb86fb015b4ff101f821736903ec56cae` | Public known-site redocking; top-1 recovered |
| `experiment-results/redock-1s3v-20260723-final/result.json` | `0a48d181dff83cfd5b60b013eb13e48314fead6704cef64f6a3ca6f99829acdb` | Public known-site redocking; top-5 only |
| `experiment-results/redock-1uou-20260723-final/result.json` | `01b8f78ba9f157c460b4f361920d1e619a8f6f2e500a1b2ada92f788512464b0` | Fail-closed unspecified stereochemistry receipt |
| `experiment-results/redock-1ke5-ls1-20260723-final/result.json` | `24594c9db67fd5873bb8413826b9534843d03e9acb5490de19c38033c5e912ff` | Fail-closed missing receptor side chains receipt |
| `experiment-results/redock-1s3v-tqd-20260723-final/result.json` | `9e97875a686179bd8c208ec2cf903e9a30c7a2e28ca90df826ed35d78db1a7c1` | Fail-closed retained sulfate receipt |
| `refine-logs/EXPERIMENT_RESULTS_20260723_VINA_FIRST.md` | `d830fb392ad0e49915fd9d9bfc232ce834c02ef7177dd7357bec72563bda3718` | Timestamped Vina-first implementation report |
| `refine-logs/EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA.md` | `bc3ac86902a5b3398b66b6b1d0fd291198b034c0fd8ca97dc299fef1aa270d93` | Historical v1 automatic-selection/direct-adapter smoke report |
| `refine-logs/EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA_V2.md` | `1169bce598890a822b0addf7e7a01b38d81ca89a46cbf0b251c6bb6b2a2fcc52` | Historical timestamped v2 report: direct adapter PASS plus host-blocked production attempt |
| `refine-logs/EXPERIMENT_RESULTS_20260723_AIAA_QUICK_VINA_V3.md` | `19baac13ae6cab3d30f54385ef8a511c46d602e993f083fda9a169ee753e4dc2` | Current timestamped v3 report: box-bound direct PASS plus host-blocked production isolation |
| `README.md` | `5c44325fb47da2892cfb03a65547fd3b109473910da29c9e6ed3db183a963a2d` | Current runnable overview with English judge quick start, fixed-ten boundaries, frozen TriPharm aggregate, optional DrutAI and experimental-data commands |
| `DOCs/PROTBIND_LIBRARY_AND_EXTERNAL_PREDICTORS.md` | `71a2d544dda01e27f318a3e11434071ad63e759d8fef2aad7e073abf6c155d99` | External-predictor admission boundary, including optional GPL-weight DrutAI acquisition, Snap isolation audit and supportive-only interpretation |
| `.agents/skills/protbind-research/references/tool-contracts.md` | `7513ce85286b4b65fb0158061448862901dd6d1dcbe30c61626f5968fce5cdd4` | Agent-facing fixed tool contracts for DrutAI and append-only experimental data operations |
| `src/protbind_agent/drutai.py` | `d72757ab52cd880567f2133e4dc52c19fafa071cfc7d0bd3e0b7ff80388fdc45` | MIT-licensed optional isolated DrutAI adapter with pinned acquisition, hash verification, Snap audit and supportive-only annotations |
| `LICENSES/ProtBind-DrutAI-adapter-MIT.txt` | `355d3fb6c1e29adef194f9f48cdfe403d84ef5fc54f4057bc9416537e94a6f4e` | File-scoped MIT grant for the independently handwritten adapter; explicitly excludes weights and upstream artifacts |
| `src/protbind_agent/experimental_assays.py` | `7b5781e0063bfd339f20c421db0767bfd26451dc3cb194221680fa58ac38a0dc` | Append-only experimental import/catalog and explicit deterministic curve-fit substrate |
| `.agents/skills/protbind-drutai/SKILL.md` | `7b835331f2361a9737570888b7606073bbe763782ba2694413f35d47ce10242b` | DrutAI operation skill preserving license, approval, runtime-isolation and scientific-claim boundaries |
| `.agents/skills/protbind-assay-data/SKILL.md` | `fb324413e1c583eb279fbfa4da0511914ffa8f3cb3c57030fcd2e707a31bd252` | Experimental-data skill for approved import, listing and deterministic fitting |
| `tests/test_drutai.py` | `7d97051a259a7993896686cb032a058e95ecfab74dfe0432af0aa67cfdc0363f` | DrutAI acquisition, identity validation, output mapping, Snap isolation and fail-closed tests |
| `refine-logs/EXPERIMENT_RESULTS_20260803_DRUTAI_SNAP_SMOKE.md` | `efec19a86db70c496f765c2eb18c2b7f6c4131c24bd08f2ebd5ab511f67db2cd` | One-record synthetic Snap runtime smoke with strict-confinement and interface-audit receipts; no biological claim |
| `submission/PROJECT_SPECIFICATION.md` | `8de77d61de4e66abdd64bd4262d570961ec5e580557fa297cd741d62616e0d47` | English Track-2 specification source with architecture, Radeon evidence, scientific limits and deployment plan |
| `submission/PROJECT_SPECIFICATION.pdf` | `2c7835df9029de37db0c0aa560d644c95301f17a241db05975dd1b99c7e0bc6a` | Six-page visually inspected Project Specification PDF with generic ProtBind Team metadata |
| `submission/ProtBind_Pitch_Deck.pptx` | `302e6dc6e4335a4569db520ac915726e7172512ca882b4f90bb0ad0676a50aa1` | Nine-slide editable and visually inspected English pitch deck |
| `submission/REPRODUCIBILITY.md` | `a9e5aa3e2e5aceb318be5febf319ffd09d47ea99b6bbdeb16a53484b08d8d4d2` | English environment, HIP build, test, smoke and local-Agent reproduction guide |
| `submission/CLAIM_MATRIX.md` | `a7334ce69f07c0496344bdf121089374d738c23f4f1a537cfa23c662ac0cfd87` | Competition-facing supported/unsupported/not-evaluated claim matrix |
| `submission/DEMO_SCRIPT.md` | `8969c27ed51340d90b4272369dbb655c36ef5036e459a70073471c7287794337` | Four-and-a-half-minute Radeon execution shot list and privacy checklist |
| `submission/PITCH_DECK.md` | `4bcb5399a18ee1e1d2e4c614c3e232840e32381ab3758b5297e521a12eb333fd` | Editable English pitch-deck source |
| `submission/PR_BODY.md` | `8fa8d052ad799a784e5ab14274c273487b9885a45014afe9462c2a272f6ca620` | English official-PR body template with explicit human placeholders |
| `submission/SUBMISSION_CHECKLIST.md` | `1bc9a607675d235d15b305d3787abde23b52cd8a845901ba7cb3ef73a763bd7d` | Deadline-oriented eligibility, deliverable, PR, evidence and licensing checklist |
| `submission/GAP_AUDIT.md` | `e238f706ef6334adfd7867e3b03f795e7793e51ea18c5112d9cdc0cdd0e449dd` | Prioritized clean-commit, video, eligibility, IP, Agent A/B and non-goal audit |
| `figures/protbind-system-architecture.mmd` | `d4ae5c28d9e1efa166215244dec1c4d2be66a9d15ead4d73a5159f26c38dfba1` | Verified full architecture Mermaid source |
| `figures/protbind-system-architecture.md` | `317f976f0da6c360b68760e1e4eb5fd62b6f28e6b8b3be5a9a073bffe737b36f` | Markdown preview of the full architecture diagram |
| `figures/protbind-system-architecture.png` | `1c43f15aec914b37999b54039d7e08bfe1fd95441d3f6c6d5d8a9419965ac609` | Visually inspected portrait architecture used in the specification |
| `figures/protbind-pitch-architecture.mmd` | `dd38694e13466e56c73b5793f9a4876d5ae7c7453ee959d663152a38b15d0c4f` | Verified compact judge-facing Mermaid source |
| `figures/protbind-pitch-architecture.md` | `2c7795d74303ab515d6f0c01338c668b59095ad546d59d1d065d0ca385642bd5` | Markdown preview of the compact pitch architecture |
| `figures/protbind-pitch-architecture.png` | `2d154a4b3a8d83383f723d9ff29e268932b787c024a77184681676eb0937d8db` | Visually inspected landscape architecture used in the pitch deck |
| `tests/test_experimental_assays.py` | `fe1d591afdd593279f2753d6860ed497319deae789239e0dede8e3a1ac0e0b42` | Experimental preview/commit, append-only catalog and curve-fit tests |
| `opencode.json` | `024c2744b18d6d1fa825709b8664101b56761a932f817b4693be8ec718114aa4` | OpenCode permission registration for the two new skills and sensitive tools |
| `DOCs/PROTBIND_AGENT_WORKFLOW.md` | `ed6b7f4d0df0b4a87206efd8c9a56db473fd07dd1065a32813ab6ec7922fc5d7` | Current deterministic Agent workflow and restrained-side-chain validation semantics |
| `DOCs/PROTBIND_PRIVATE_RESEARCH_AGENT_PLAN.md` | `d9f8da4765fd11de0c8fb868dc5b95890131f3adc7b6df38271c23eb3176f54d` | Current implementation/acceptance plan with original, v1 and v2 result boundaries |
| `refine-logs/EXPERIMENT_RESULTS_20260725_SCIENTIFIC_GAPS.md` | `2cc5dec238a1dc36e2a2374cfb9a0d69008b8ea450c2365da71ada92102d0b86` | Timestamped fixed-ten, repair, calibration and v4 scientific-gap report |
| `refine-logs/EXPERIMENT_RESULTS_20260726_REPAIR_PROTOCOL_V1.md` | `5f988950f0ac1aeb1d1400cb2455c6d14d857b8697979ee934a2c2774485bbbe` | Timestamped revision freeze, per-case metrics, repair receipts and claim boundary |
| `refine-logs/EXPERIMENT_RESULTS_20260727_RESTRAINED_SIDECHAIN_V2.md` | `000aca5255fd3a9b2264f9f3cedc9a4adf020a4337a2944457b4c874745702bd` | Timestamped v2 protocol, per-case results, constrained-geometry receipts and claim boundary |
| `refine-logs/EXPERIMENT_RESULTS.md` | `2b963752be26b51e8f57a57ee4a70d6df061bda931f917556a7c72ffc242db1c` | Current concise handoff; original 7/10, v1 8/10 and v2 9/10 are separate |
| `refine-logs/EXPERIMENT_PLAN_20260802_TRIPHARM_THREE_GATES.md` | `f5173446cc07b847d2ab7baf935590cd637550219ead7c3f2d241baced078331` | Timestamped claim-driven plan for prospective science, fair baselines, and persistent HIP acceleration |
| `refine-logs/EXPERIMENT_PLAN.md` | `7cda3714a167698a4e9608523cb5faf2f3ad4ed4f1a8f409ec86ba7f4149908c` | Stable entry point for the current TriPharm three-gate plan |
| `refine-logs/EXPERIMENT_TRACKER_20260802_TRIPHARM_THREE_GATES.md` | `67e06222b6d492e24193f4942130dbef406166c693774afc7fdf274f090389de` | Completed three-target tracker; break-even prefixes and R9700 remain open |
| `refine-logs/EXPERIMENT_TRACKER.md` | `9e738302a04e6d22b763e37252df411168166fc2f1f6fb8719678733486c1741` | Current global tracker with completed frozen TriPharm validation, Agent/DrutAI/assay status and 454-test QA |
| `DOCs/PROTBIND_PHARMER_THREE_WAY_BENCHMARK.md` | `665b0d34bd83c7c2a5d1b6a35186765b505c8f2c48558547a8e2f97f671b2250` | Pharmer/CPU/HIP results, bounded claims and 30-repeat static prefilter timings |
| `experiment-results/pharmacophore-three-way-20260802/prospective-three-target-aggregate-v1.json` | `6dd425a9f9bba8a151fa6d6ac80b52e18ffac614776351c33a31d134eee77394` | Self-bound three-target aggregate; internal aggregate SHA is `5ac3a9ca...362d` |
| `scripts/summarize-prospective-three-way.py` | `6ef67e61e87b15bad0e2600dcc1585bc0563257ef7ae6b00dbbafc7ede815403` | Deterministic aggregate builder and frozen gate evaluator |
| `scripts/benchmark-tripharm-static-kernel.py` | `15e8913ba2bb7e8f5adaf4ade58a5b275184855eb3f606b0e26d0d5da57f9064` | Static HIP prefilter 5-warmup/30-repeat benchmark runner |

Runtime identities:

- OpenFold source allowlist SHA-256:
  `742e9bf654b13f67783d095a2327af3ed31163580eaa7b4c548e8a8eb2e68010`.
- AutoDock Vina binary SHA-256:
  `f31f774f723bba7bbe6e9d1c47577020eea9a8da16424284c043d22593570644`.
- Three-complex redocking source-manifest SHA-256 (30 `src/protbind_agent/**/*.py` files):
  `ed0f8820500ca2dca5fd3c2a0cc616148d5da39197cc48e898667de853559f45`.
- Three-complex redocking toolchain SHA-256:
  `545751d2e8a8cc816e65f8828a50acaf32a2207f77a9556c33230ade7d302f86`.
- Formal fixed-ten ProtBind code/config SHA-256:
  `5b2a4fbf61d827379bc464c8f65a9664e146d41b482e1dd41cf39490317363ce` /
  `756851f4f3173c1b97ad7fb4b3bba275eb7c333718c8d12b31746a5d0b1b6a87`.
- Retrospective repair code/config SHA-256:
  `71499cfceae09c865dedd07b4ea2c9ea9b793738eaf632d1b734d505554ec543` /
  `f15e26699b3b3f8b1be0c6afefa690dee6c1b86927e717c656cc0c432fc0bdc9`.
- `repair-protocol-v1` code/run-plan/batch/regression internal SHA-256:
  `0fcb6c2bad48da233b0b4a543cd88e32b6eac5d5fff7c49d8c93b02e0f223741` /
  `bb714243c4ec860694dde0d10f0182b7c48a192d6fcdac42be0672fd787b903f` /
  `7b014928c0688f1a64a661b153e0c846ee2a50ef181eab8900dd48714d07a5bf` /
  `076643a00d761dc00afc9a81e64ca69c509eba28743e49c8ad0b786e51459abc`.
- `repair-protocol-v2-restrained-sidechain` code/run-plan/batch/regression internal SHA-256:
  `56cc89f42e8397340b64546bc1824adbe80730eab0b5ae1a632fbf2d1c718e5d` /
  `26bf03ed973b18bfde4c3f5af4b58da5d5d2e5131b83f20795df95375af18e68` /
  `566128395b794b488cfbc5153b0b599a0b5676eb3d532d6d4eee3918a39a0692` /
  `5f524930248aebe717a316a0a44c89be95e805b354ef5b00eaaf7436d3c70b35`.
- Restrained-side-chain implementation SHA-256:
  `preparation.py` `ccc12e039c41cd8750761a5526a0c960b0756d841b1539834a1a16d2383795e8`,
  `redock_benchmark.py` `b827cd236fd22ff14ea20e0b50b41d261a6885923de4a8de2d0d24c0b0323e8d`,
  `redock_holdout_batch.py` `754116608015f4616bea8f76cf1683e97c3ec00bc857c41dcca11810864fadc0`,
  `cli.py` `1ba67c1acfba526bfe5cb62ae08e41723769c733117a8601f5adaaaa92c97b43`.
- Restrained-side-chain tests SHA-256:
  `test_receptor_preparation.py` `60d0ba4dc495ecdf1ce22502a915d82c05abe30be84a3caf2de9ffd9189e4697`,
  `test_redock_benchmark.py` `79ba8bed11d15ec069178b69cae8fdd050fead61b6fb590233e9cd34087a7592`,
  `test_redock_holdout_batch.py` `f3f6df8557d8c5d2ad3d583bdd6df524930a2c5a1d5e01c5d19f19f9a401f83f`,
  `test_protbind_cli.py` `45c4efa6179dde221bb698bbd0effda9cf0263dbd35a7c99dae1bda441e5b5fe`.
- Repair revision binding implementation/test file SHA-256:
  `redock_holdout_batch.py` `4b1a1db5181303b72b77fdbb193ff2b06bb1035b959f591a76b439875ae9f71d`,
  `cli.py` `dcf9bac2c6f18e9be1ebb82387a2509514cedb67c051c18ab12ae6c45cfaf650`,
  `test_redock_holdout_batch.py` `40ecb8bbe6c8960272fd05b325316c395a5a8fb040ddb6ea3cf10a637fbae365`.
- RCSB 1CRN raw mmCIF SHA-256:
  `23787562c427d7c1abe5420e86d5f1d0a6c7007dec1e8ce85645a6d69c32e8ba`.
- RCSB-selected receptor SHA-256:
  `557a87ba4c43c38c8b245617d78f3c00a73100dafe451e59fcfb806482c2d366`.
