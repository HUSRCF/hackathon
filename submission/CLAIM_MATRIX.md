# Submission claim and evidence matrix

| Claim | Status | Evidence | Explicit boundary |
|---|---|---|---|
| Local Agent completes the fixed research tool sequence on W7900 | Supported | `protbind-agent-w7900-c58ca3c.json`; 3/3 tool and citation success | One host, one frozen model/runtime configuration |
| Core Agent inference runs locally on AMD Radeon/ROCm | Supported | Bound W7900/gfx1100, HipFire daemon, model and loaded HIP runtime receipt | Not a cloud API result |
| TriPharm HIP output matches CPU reference | Supported | Complete CPU/HIP score-vector parity on ALDH1, MAPK1 and MTORC1 | Parity is not biological correctness |
| TriPharm provides end-to-end application acceleration | Not supported | CPU exact finalizer dominates current full path | Report kernel/prefilter timings separately |
| Latest docking protocol recovers 9/10 fixed cases | Supported as controlled revision | Frozen v2 run plan and independent regression | Same ten cases were previously observed; not prospective generalization |
| ProtBind predicts experimental affinity | Not evaluated | No prospective affinity experiment | Vina, DrutAI and IFP are not affinity evidence |
| TriPharm broadly outperforms Pharmer | Not supported | Frozen three-target aggregate | Target-dependent; Pharmer wins ALDH1 EF1% |
| DrutAI independently verifies binding | Not supported | Sequence–SMILES annotation only | No pocket, pose, calibration or experimental evidence |
| Private sensitive tools require explicit approval | Supported | Tool contracts, permission tests and approval timeline | Cross-process pending persistence remains future work |
| Experimental tables are immutable and auditable | Supported for tabular P0 | Preview/commit tests and content-addressed receipts | Specialized image analysis and assay-specific kinetics remain future work |
| R9700/gfx1201 produces cross-architecture parity | Not evaluated | Device not yet run | Do not imply from gfx1100 results |
