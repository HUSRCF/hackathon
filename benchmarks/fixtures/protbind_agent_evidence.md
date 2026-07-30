# ProtBind Agent benchmark evidence fixture

## Scientific boundary

This public fixture tests local retrieval and Agent tool orchestration only. TriPharm scores are
geometric pharmacophore matches, AutoDock Vina values are pose-ranking tool scores rather than
experimental binding free energies, and a model-generated pose remains a hypothesis. A visual
overlay is quality assurance and cannot replace PoseBusters, symmetry-aware RMSD, ProLIF, or a
reviewed reference structure.

## Privacy boundary

The benchmark is fully local. Private protein sequences must not be uploaded. Every reported
scientific statement must cite an immutable `sha256:` artifact or a document section/page.
