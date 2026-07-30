# Radeon Agent Lab

面向 AMD AI DevMaster Hackathon **Track 2: Agentic AI** 的本地私有 Agent 骨架。计划中的最终部署与
验收矩阵为 1/2/4×`gfx1100`；当前主机是 2×Radeon Pro W7900，但现有折叠 adapter 仍是单
GPU、单 job 路径，尚未实现或验证多 GPU adapter 调度。`gfx1201` 只保留为有设备时的可选交叉检查，
不阻塞比赛提交。

这一版先固定四件事：Agent 与推理引擎解耦、工具权限显式化、硬件证据可追溯、跨架构结果可比较。
HipFire 作为外部推理服务接入，不复制、修改或嵌入其实现。

## ProtBind 私有科研工作流

仓库现在同时包含 `protbind`：一个以工具证据为边界、可断点恢复的蛋白质—配体科研工作流。它不会让
LLM 生成结构、对接分数或 ADMET 数字；缺少主路径所需的 Vina/PoseBusters 等科学能力时，运行会
明确进入 `DEGRADED`，并保留最后成功阶段和原因，而不是生成占位结果。OpenFold3 是可选旁路，未配置
或失败不会阻止 Vina、验证和报告。

当前可直接运行的 P0 基础层包括：

- `ResearchCase` 三模式 schema、v1 体系边界和隐私策略；
- SHA-256 内容寻址 artifact、hash-bound manifest、原子写入和恢复审计；
- TriPharm CPU reference（六类特征、持久化 SQLite warm index、三点对应、Kabsch/Horn overlay、
  确定性排序）及 `both` 模式等权 RRF；
- 可选 RDKit SDF/SMILES/CSV/Parquet 导入，以及 Gemmi 结构检查/启发式口袋药效团；
- 蛋白折叠前的结构拦截层：用户结构 → 本地精确序列缓存 → 显式授权 RCSB 导入 → 折叠兜底；
- 独立科学环境的 JSON worker 协议、schema-2 主状态机、确定性 scaffold/microstate/quick-Vina
  selection builder、证据等级和 seekdb/BGE-M3 capability gate；
- 固定到 OpenFold3 0.4.3 指定 commit 的断网 CLI adapter；官方源码已在复用 AIAA Torch/ROCm/Triton
  的轻量 overlay 中通过官方 ROCm/Evoformer validator，但尚未导入 checkpoint 或运行科学 bake-off；
- CPU-only、运行时 hash-attested 的 Meeko/Vina adapter；规范 docked pose 为 Meeko 恢复的 SDF，
  同时保留最佳/全部 PDBQT、全部 modes SDF、pose-extraction receipt 和 receptor-preparation receipt；
- 从精确 docking bundle 自动构建验证 batch/toolchain 的适配层，以及强制 PoseBusters、可选
  sPyRMSD/ProLIF/OpenMM 的验证 adapter；result-blind fixed-ten 已冻结并尝试 10/10，8 例完成、
  2 例受体制备 fail closed，正式 top-1/top-5 均为 7/10；独立 ProLIF 使用 receipted 8 Å
  whole-residue crop，保守重原子修复只允许发生在受保护口袋之外；
- fair-esm ESMFold v1 三权重、环境 lock 与运行时代码复合哈希、restricted-unpickler、输出结构 QC
  和单 GPU lease；已在一张 W7900 上完成短序列断网烟测，但不把该结果写成结构准确率或官方
  checkpoint 等价性证据；
- FastAPI 本地六页面和一个真实可编译的 HIP triangle-matcher microbenchmark。

先检查环境：

```bash
PYTHONPATH=src python -m protbind_agent doctor
```

生产前必须检查 `runtime_details.worker_network_isolation`：其 `status` 只可能是
`missing | present_but_unusable | usable`。当前宿主实测为 `present_but_unusable`，bubblewrap probe
`return_code=1`；`offline_default` 和 application offline 环境变量只是应用策略，不等于 OS 网络隔离。

使用预计算药效团 JSONL 建索引（不需要 RDKit）：

```bash
protbind index build \
  --input examples/library.features.jsonl \
  --output artifacts/protbind/library.sqlite
```

case JSON 可使用 `target.structure_file`、`ligand.pharmacophore_file` 和
`pocket.pharmacophore_file`；CLI 会先导入私有 artifact，manifest 和报告不会保存这些文件的绝对路径。

```bash
protbind case run \
  --case examples/case.json \
  --index artifacts/protbind/library.sqlite \
  --mode ligand_only \
  --stop-after screened
```

若 case 的 `target` 提供 `pdb_id` 或 `uniprot_accession`，可在折叠前尝试 RCSB。显式 PDB 只需批准
坐标下载域；UniProt 发现还需批准 Search API。两者都不会上传蛋白序列：

```bash
protbind case run --case examples/case.json --index artifacts/protbind/library.sqlite \
  --rcsb-pdb-id 4HHB --rcsb-chain A \
  --approve-network files.rcsb.org --stop-after screened

# Biological assembly is never inferred: request an assembly ID explicitly.
protbind case run --case examples/case.json --index artifacts/protbind/library.sqlite \
  --rcsb-pdb-id 1ABC --rcsb-assembly-id 1 \
  --approve-network files.rcsb.org --stop-after screened

protbind case run --case examples/case.json --index artifacts/protbind/library.sqlite \
  --rcsb-uniprot-accession P69905 \
  --approve-network search.rcsb.org --approve-network files.rcsb.org \
  --stop-after screened
```

没有 PDB/UniProt 标识时，序列检索必须额外传 `--approve-sequence-upload`；普通网络批准不能代替该
门禁。默认坐标策略明确为 deposited asymmetric unit；只有 `--rcsb-assembly-id` 才下载对应的
biological assembly，二者使用不同 URL 和缓存键。远端候选只有在链分配唯一、坐标序列精确匹配、
N/CA/C 完整、坐标有限、无未解析 altloc 且无金属/声明的共价连接时才会进入本地缓存。
多链条目可按显式 chain ID 或唯一精确序列抽取，但所有保留/丢弃 chain ID 都写入
`protbind.structure-resolution` receipt；显式 chain ID 也是缓存身份的一部分，不能用同序列的另一条
chain 命中。用户结构、本地缓存和 RCSB raw 均经过统一连接检查：mmCIF 检查 `_struct_conn`，PDB
检查 LINK 及标准蛋白—非标准配体 CONECT；PDB 中“未发现声明”只记录为部分证据，不表述为完整
非共价证明。原始下载 mmCIF 和抽取后的 receptor 分别保存为 SHA-256
artifact，并共同绑定到 run manifest；HTTP transport 忽略环境代理，精确域名批准不会被代理静默改道。
这里的 raw artifact 指坐标 mmCIF；当前尚不冻结 RCSB Search 原始 JSON 或 request body/hash，receipt
只记录请求类别、是否上传序列及实际尝试的候选，因此不得把它描述成完整 Search 响应审计。
在任何获批网络请求前，workflow 会先冻结并解析 index、药效团、SMILES/配体化学与已有本地 artifact；
损坏的精确缓存也会 fail closed，不能因本地错误继续上传序列。歧义、错序列和金属/声明共价结构只会
触发折叠兜底或显式失败，不会静默接纳。

独立的 public-data acquisition 层可在创建 case 前按公开标识符获取候选蛋白/小分子文件。它只允许
固定 registry 和构造出的 HTTPS URL，要求精确域名批准，并以 direct curl 禁用环境代理与重定向；
下载物、远端元数据、校验结果和 receipt 都内容寻址。输出路径及来源规定的后缀会在联网前检查：

```bash
# 实验结构候选；默认同时运行本地 PROPKA 审计
protbind data fetch --source rcsb-mmcif --identifier 1CRN \
  --output inputs/rcsb-1crn.cif --project-root . \
  --approve-network files.rcsb.org

# AlphaFold DB 预测结构候选
protbind data fetch --source alphafold-mmcif --identifier P69905 \
  --output inputs/afdb-P69905.cif --project-root . \
  --approve-network alphafold.ebi.ac.uk

# 蛋白序列、CCD ideal coordinates、PubChem computed 3D record
protbind data fetch --source uniprot-fasta --identifier P69905 \
  --output inputs/uniprot-P69905.fasta --project-root . \
  --approve-network rest.uniprot.org
protbind data fetch --source rcsb-ccd-sdf --identifier HEM \
  --output inputs/HEM-ideal.sdf --project-root . \
  --approve-network files.rcsb.org
protbind data fetch --source pubchem-cid-sdf-3d --identifier 2244 \
  --output inputs/pubchem-2244-3d.sdf --project-root . \
  --approve-network pubchem.ncbi.nlm.nih.gov
```

Gemmi 对 mmCIF 做确定性解析，并报告声明的未观测残基/原子、标准残基骨架/羰基缺失、altloc 和
零占有率；RDKit 检查单分子 SDF、3D、重原子、碎片、金属和未分配手性。蛋白结构默认再通过当前
AIAA Python 的 `python -m propka` 生成 pKa/质子化可行性及诊断 artifact；可用
`--skip-propka` 明确跳过。PROPKA 成功不证明结构完整、位点正确或可对接，下载物也不会自动附加
或推进 case，仍须经过正常的身份、链、金属、共价与 receptor/ligand 门禁。PubChem 汇聚记录没有
统一许可证断言，receipt 会要求在再分发前复核 contributor provenance。

当前 ESMFold v1 是外部 attested worker + receipt 导入路径，不会在 `RECEPTOR_READY` 内自动启动。
先把无结构 case 停在 `INPUT_VALIDATED`，再于 `RECEPTOR_READY` 前附加与目标序列精确一致的 receipt；
generic `esmfold_structure` attach 会被拒绝：

```bash
protbind case run --case case.json --index library.sqlite --stop-after input_validated
protbind case attach <run-id> --name esmfold_receipt \
  --file <private-receipt.json> --media-type application/json
protbind case resume <run-id> --worker-config configs/protbind-workers.toml
```

最终 24 aa 单 W7900 warm smoke receipt 记录 8,496,247,808 peak allocated bytes、26.112 s model
load、3.653 s inference 和 37.425 s end to end；它只证明本地断网协议/环境可执行，不是 700 aa
显存保证、结构准确率或性能 benchmark。自动 receptor resolver 当前仍只有 RCSB；AlphaFold DB
已经能按 accession 获取本地候选结构，但尚未自动进入 resolver，ESMFold2 仍为 future-only。

schema 2 的主状态固定为：

```text
CREATED → INPUT_VALIDATED → RECEPTOR_READY → INDEXED → SCREENED
→ SELECTED → DOCKED → VALIDATED → REPORTED
```

`COFOLDED` 只保留为 schema-1 只读 manifest 的历史状态和 schema-2 的可选 worker task 名。可选共折叠
状态独立记录为 `NOT_REQUESTED | UNAVAILABLE | RUNNING | COMPLETED | FAILED_RECOVERABLE`，不会成为
Vina 的上游依赖。

### 交互式逐阶段闭环

Agent 路径不会直接调用 `case run/resume` 从头跑到底。`StageGateController` 在每个主阶段前深度
复核 manifest、配置和所有已完成 artifact，签发绑定当前 manifest SHA-256、下一阶段和控制策略
SHA-256 的一次性 continuation token；每次只能推进一个主阶段。阶段结束后再次复核 stage record、
cache binding 和输出闭包，并生成内容寻址 acceptance receipt。只有 `ACCEPTED` 才会开放下一阶段。

```bash
# 人工检查/推进同样使用闭环控制器
protbind case gate <run-id>
protbind case advance <run-id> --continuation-token <fresh-token>

# OpenCode 通过本地 stdio 启动同一受限 MCP 服务
scripts/aiaa-protbind.sh -m protbind_agent mcp serve \
  --workspace artifacts/protbind --project-root . \
  --library-config .protbind/library.json \
  --knowledge-model .protbind/models/bge-m3
```

门禁决策为 `READY | NEEDS_ACTION | RETRYABLE | UNSUPPORTED | FAILED | COMPLETE`；postflight 另有
`ACCEPTED`。缺受体、query、selection batch、worker 或环境 lock 时停在 `NEEDS_ACTION`；worker
crash/OOM 等可恢复失败停在 `RETRYABLE`，但没有自动重试。陈旧 token、artifact 篡改或配置变更都会
fail closed。每个 run 的 `control.json` 只保存内容寻址 gate/acceptance receipt 引用。

仓库根目录的 [`opencode.json`](opencode.json) 只启用回环 HipFire provider 和 `protbind` stdio
MCP，默认拒绝所有工具；case 只读状态/报告可直接调用，创建、附件和单阶段推进均要求交互批准；
私有 PDF、protein/ligand library 连读取也始终为 `ask`，且工具参数还要求一次新的显式确认。项目 skills
位于 [`.agents/skills/protbind-research/SKILL.md`](.agents/skills/protbind-research/SKILL.md)
和 [`.agents/skills/protbind-library/SKILL.md`](.agents/skills/protbind-library/SKILL.md)，
OpenCode 可原生发现。完整 CLI、迁移/验证状态、P2Rank 与 DrutAI 门禁见
[私有数据仓库与外部预测器说明](DOCs/PROTBIND_LIBRARY_AND_EXTERNAL_PREDICTORS.md)。
PDF/OCR 门禁、引用语义和蛋白库 RAG 模型选择见
[PDF 与蛋白库 RAG 说明](DOCs/PROTBIND_PDF_AND_LIBRARY_RAG.md)。
配置未写入不存在的 production worker digest；因此默认会诚实运行到能力门并
停住，生产时须由操作者在 MCP command 中加入已审核的
`--worker-config configs/protbind-workers.toml`。当前精确白名单的 23 个受限工具已完成 MCP 1.14
stdio 真实 initialize/list-tools 握手与权限回归。普通 public-data fetch 与 accession-only
UniProt 验证是仅有的受控网络表面，
只接受白名单 source/公开 ID/精确域名批准；`case_dossier` 与 `case_pose_view` 仍为只读、无坐标工具。本机 OpenCode 1.18.8
已完成 DeepSeek V4 Flash 的无工具云端传输冒烟测试。HipFire TUI 端到端对话仍须在本地模型服务
启动后单独验收。

OpenCode 1.18.8 本身没有列出 PDF 文字提取或 OCR 内建工具；ProtBind 因而没有开放通用 bash，
而是提供受限的 `knowledge_document_inspect/import/search`。PDF 逐页先比较 PyMuPDF 与本机
`pdftotext -layout`，标出低文字量扫描页；OCR 可设 `off|auto|required`，缺 Tesseract 时
`required` 会 fail closed。每次导入生成含工具版本、逐页 backend、未解析页和页码引用策略的
内容寻址回执。

### 运行 dossier 与本地姿态查看器

`case report` 是最终科学结论报告；`case dossier` 是任意时点的执行账本，逐阶段列出计算完成、
postflight acceptance、耗时、输入/配置/cache hash、输出、warning、失败和 artifact inventory。
二者不可互换：一个阶段有 stage record 但没有 `ACCEPTED` receipt 时，dossier 会明确显示
`COMPLETED_UNRECEIPTED`，不会假装闭环完成。

```bash
protbind case dossier <run-id> --format markdown
protbind case poses <run-id>

# 仅首次预取；精确批准域名，固定 3Dmol.js 2.5.4 并校验 SHA-256/许可证
protbind assets install-3dmol \
  --approve-network cdn.jsdelivr.net \
  --workspace artifacts/protbind

# 此后可拔网运行，只监听回环地址
protbind serve --workspace artifacts/protbind
# 浏览 http://127.0.0.1:8765/runs/<run-id>/poses
```

也可以用 `--from-file <3Dmol-min.js> --license-file <LICENSE>` 完全离线安装；文件仍必须与冻结哈希
一致。浏览器只从 workspace 的已验证 asset 加载，不回退 CDN。受体/配体坐标只由 loopback viewer
按所选 candidate 读取；MCP/Agent 只看到 artifact 引用、Vina 工具分数、验证结果、box 和
Gemmi/RDKit 生成的无坐标几何摘要。PNG、近距离 pair 计数、box containment 和人工观感仅是 QA，
不能替代 PoseBusters、ProLIF、对称 RMSD 或 OpenMM。

配置 `[workers.select]` 后，`SELECTED` 会自动从 `SCREENED` 生成 Bemis–Murcko top-128、微状态和
typed `protbind.quick-vina-input`，用独立的 CPU-only `vina-quick` profile 完整评估计划请求，再按真实
工具分数冻结 top-16 `selection_bundle`。quick profile 固定为裁剪证据，不能替代 `DOCKED`；后者会用
更高 exhaustiveness 独立重跑 Vina。初次运行可同时冻结 AIAA/Vina environment lock：

```bash
protbind case run --case case.json --index library.sqlite \
  --worker-config configs/protbind-workers.toml \
  --vina-environment-lock experiment-results/aiaa-environment.json
```

自动路径 v1 只接受 case 中明确的 pocket `center + box_size`。只有残基、只有配体或没有明确 box 时，
在 fpocket/P2Rank 尚未配置的情况下返回 `DEGRADED/SITE_DISCOVERY_UNAVAILABLE`，绝不猜测全蛋白 box。
当前 selection preparation/finalizer 为 2.5、quick input producer 为 1.2、quick worker profile 为
1.3，并强制生成 schema/producer 2.0 的 `protbind.docking-box-receipt`。receipt 绑定 receptor 的完整
`ArtifactRef`/SHA-256、box 来源、center/size 和 `receptor-cartesian-angstrom` 坐标系。每个维度必须在
4–60 Å，体积不得超过 27,000 Å³；它还会从精确 receptor artifact 重新解析并确认 box 内至少有一个
标准蛋白重原子。同一 receipt 必须贯穿 selection preparation、quick input、每个 request、内外层
evidence、evaluation batch/run metadata 及最终 selection bundle。

原子重叠只证明 box 和 receptor 的坐标系看起来相容，不证明它是生物学结合位点。`user-center` 和
`user-residues` 始终按 `user-hypothesis-only` 解释；`co-crystal-ligand`、
`fpocket-p2rank-consensus` 和 `public-benchmark-reference` 必须另附 hash-bound、
coordinate-free 的 `protbind.site-derivation-evidence`，绑定 receptor、center/size、坐标系、推导方法
及非空 source commitment SHA-256，并明确不把验证参考坐标暴露给筛选侧。该证据只验证推导来源，
仍不推断位点本身为真实生物学事实。

`both` 模式可额外声明 target-specific known-site calibration receipt。selection 2.5 会在任何
manual/automatic selection 前重新解析它绑定的 canonical source redock、Meeko prepared receptor、
receptor-preparation receipt 与 native-derived box，并核对 PB-valid/symmetry-RMSD 门；target、
receptor 或 box 任一不一致均 fail closed。它只授权 known-site box，不把 native ligand
identity/coordinates 暴露给 quick worker，也不构成结合或亲和力证据。
`case attach --name selection_batch` 仍保留为导入已审计外部 batch 的兼容/恢复入口。

2026-07-23 的历史 quick profile 1.2 / selection config 2.3 / quick input producer 1.1 直接 AIAA v3
adapter smoke 曾完成 3/3 个真实 Meeko/Vina quick 请求及 36-entry 输出闭包；三个 Vina 工具分数为
`-1.942/-1.896/-1.753`，不是实验结合自由能。输入仍是非已知结合位点的公开 1CRN、
`chemistry_verified=false` 的 demo index，并且只验证 application offline policy；所以它只证明
历史 adapter 的协议/环境路径，不是对接质量、结合、排序、吞吐、OS 隔离或 Radeon 性能证据。该
[`smoke-result.json`](experiment-results/aiaa-selection-quick-vina-20260723-v3/smoke-result.json)
SHA-256 为 `c0e077c2d8c24e59fc4f6d3eece777f1c455b5fd325a7c890152d724339c11ee`；
[`vina-provenance.json`](experiment-results/aiaa-selection-quick-vina-20260723-v3/vina-provenance.json)
SHA-256 为 `b9b226eb718c2435f7450395f1ac40c2b1ae27a42ce40b02b8a316edfbae1536`。历史 v3 沿用已冻结的
environment lock `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35` 与 runtime asset
`e78b0d4eda4f223e7275270cdde325ae07cd86c490283319b87381853a0a0dd8`；quick/full-Vina code SHA-256
分别为 `507fd3ac9d311cacd7df516e66d38043f46045d8727e6b36b58b489c8f742be9` 和
`e800f8b94c41a343582742d3a8bfbfacaa44a5fbde868f4bfcb58c7deb054334`。由于当前 site-gate 契约与
runtime composite hash 已升级，这组 profile-1.2/config-2.3/input-1.1 artifact 不能充当当前契约
证据。2026-07-25 current v4 使用 profile-1.3/config-2.5/input-1.2/box-receipt-2.0 完成 3/3 请求与
36-entry closure；box 内有 269 个蛋白重原子，但它仍是 user-center、非已知结合位点、
`chemistry_verified=false`、application-offline direct smoke。其
[`smoke-result.json`](experiment-results/aiaa-selection-quick-vina-20260725-v4/smoke-result.json)
SHA-256 为 `8b19a1d9aa76af0f60dcf0d95c17aa86357598b315d40478dd59c238da0e6f8e`，
[`vina-provenance.json`](experiment-results/aiaa-selection-quick-vina-20260725-v4/vina-provenance.json)
SHA-256 为 `b4f51a5b93e5ac676edb328d7f86b177c009c7e90a11b8a9e444d0f0e0c7739f`；quick/full code SHA-256
分别为 `ed48d5b6884bd4a572cd7315d046b557b2bfd0ea3d1953dd72ee5acfa7458d14`/
`a181906553cc0960a9e51f02856df3b2bc527fd4a0aad24321d61bd92760f56c`。它只证明当前 direct
adapter 协议路径，不证明 docking quality、binding、ranking、production isolation 或 Radeon 性能。
v2 receipt 仍保留在
[`experiment-results/aiaa-selection-quick-vina-20260723-v2`](experiment-results/aiaa-selection-quick-vina-20260723-v2)
作为历史证据，不是当前 contract；214 tests 也是 v3 历史契约的计数。

另一个历史 `chemistry_verified=true` v3 production workflow run 已完成到 `SCREENED` 并尝试启动
`SELECTED` worker；selection preparation 与 quick input 均绑定 box receipt
`641d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a`。宿主不允许 bubblewrap 配置
loopback/network namespace，返回 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`，因此
[`manifest.json`](experiment-results/aiaa-selection-quick-vina-20260723-v3/production-workspace/runs/production-isolation-smoke-v3/manifest.json)
保持 `DEGRADED`、`last_completed_stage=SCREENED`、recoverable `WORKER_CRASH` 且没有 quick 结果；其
SHA-256 为 `730e54551f807ed18896257fa3d2f47ba9da3a930c8bc2b7b8a2c6ebff585746`。这是宿主隔离能力
阻断，不是允许关闭 `isolate_network` 或启用 fixture bypass 的理由；生产门仍未通过。

如果还要运行 OpenFold3 作为 top-8 附加证据，才需要把已冻结的 OpenFold batch、checkpoint 和
environment lock 附加到同一个 run。自动 selection 时先用 `--stop-after selected` 冻结候选，再构造
与其身份一致的 top-8 batch，并在 `DOCKED` 前附加；side task 在 `SELECTED` 后运行。CLI 会自动给
`--name` 加 `support_` 前缀：

```bash
protbind case attach <run-id> --name openfold_batch \
  --file openfold-batch.json --media-type application/json
protbind case attach <run-id> --name openfold_checkpoint \
  --file openfold3.ckpt --media-type application/octet-stream
protbind case attach <run-id> --name openfold_environment_lock \
  --file experiment-results/aiaa-openfold3-environment.json \
  --media-type application/json
```

完成任何可选 OpenFold attach 后再恢复：

```bash
protbind case resume <run-id> --worker-config configs/protbind-workers.toml
```

`openfold-batch.json` 不是候选 ID 清单，而是同时绑定本 run 的 screening artifact、library
index 和允许 receptor 的 `protbind.cofold-input-batch` v1.0。它必须覆盖可选 top-8，并与
selection 中已经冻结的 parent/microstate/quick-Vina 身份一致。OpenFold3 adapter 只产生旁路
`protbind.cofold-bundle`；其缺失或失败不会把主 run 改成 `DEGRADED`。
当前仓库只有该 batch 的 fail-closed 验证器，没有把自动 selection bundle 转换为
`protbind.cofold-input-batch` 的公开 builder/CLI；所以上述步骤仍是由受信外部工具构造并审计 JSON
后的专家导入口，不是现成的自动 OpenFold 主链。

恢复进入 `DOCKED` 时，workflow 先用一个 `stage=COFOLDED`、`previous.stage=SELECTED` 且包含上述
三个 support 的 schema-2 envelope 尝试可选 OpenFold；随后正式 Vina 仍以 `SELECTED` 为主上游，
只在旁路成功时额外携带 `cofold_evidence_bundle`。状态依次为 `RUNNING → COMPLETED`，或记录
`UNAVAILABLE`/`FAILED_RECOVERABLE` 后继续 Vina。

OpenFold adapter 的 query JSON 在顶层写入 `seeds` 并为每个 query 显式禁用 main/paired/all
MSA；runner YAML 重复固定 seed 并禁用 MSA Server。Adapter 从当前隔离环境使用
`python -m openfold3.run_openfold`，不允许替换可执行路径。它可读取 batch 中的本地 direct-CIF
template，并输出带模型置信度语义的 `protbind.cofold-bundle`。checkpoint hash、adapter hash、
全部 `src/protbind_agent/**/*.py` 的 canonical path/hash manifest（`protbind_runtime_sha256`）、
环境 lock hash、安装的 OpenFold
runtime hash manifest 和 pinned commit 共同进入 provenance。OpenFold runtime 从 Python 实际 import
的 package root 扫描，而不信任 distribution `RECORD`，因此也适用于官方 editable overlay 环境。
Runtime
allowlist 包括全部 `openfold3/**/*.py`、`core/data/resources/` 下资源及
`model_setting_presets.yml`；官方 0.4.3 基线必须恰为 317 个文件，manifest SHA-256 为
`742e9bf654b13f67783d095a2327af3ed31163580eaa7b4c548e8a8eb2e68010`。
运行前还会精确验证 distribution `0.4.3`、SCM tag/distance/node/dirty 以及 entry point。完整配置
说明见 [`workers/README.md`](workers/README.md) 和
[`configs/protbind-workers.example.toml`](configs/protbind-workers.example.toml)。
生产 host 只接受 `engine=openfold3` 与 `official-openfold3` 候选，并交叉验证 bundle producer、
checkpoint/environment-lock/stage-envelope/provenance、query/runner/run-metadata、完整 raw-output inventory
以及精确 returned-artifact 集合；engine 拼写或自报替代引擎不能绕过全局 lease 和严格 mmCIF 门。

Worker argv 必须从绝对路径调用 `scripts/aiaa-openfold3.sh`，显式进入继承 AIAA
Torch/ROCm/Triton 的专用 overlay；不允许由当前目录暗中选环境，也不允许再安装第二份 Torch。
运行时附加 `aiaa-openfold3-environment.json`，绑定基座版本、官方 validator 和源码 allowlist。
OpenFold 资源策略是每个 job 只暴露一个规范数字 `HIP_VISIBLE_DEVICES` index，拒绝同时设置
ROCr/CUDA mask alias，且一次只运行一个 OpenFold job。上游 OpenFold 的多 device 模式也只是分发
独立 query、不汇聚显存；当前 ProtBind adapter 不暴露该模式。双 `gfx1100` host 将 GPU0 用作科学
计算 lane（OpenFold 完成后才可接 OpenMM），GPU1 留给 HipFire。Runner 强制 FP32 (`32-true`)。
`checkpoint_name` 固定为 OpenFold3 0.4.3
文档中的 `openfold3-p2-155k`，官方字节数必须为 2,287,928,196 bytes（约 2.13 GiB）。当前没有
受支持的小/中/大 checkpoint 系列。
checkpoint 后缀和文件大小都不是模型显存档位，也不能用来推断推理峰值显存。
当前上游未在这份仓库中提供/冻结可独立信任的 checkpoint SHA-256 allowlist；本地
`weight_sha256` 只能证明该 run 使用了哪一份文件，不能仅凭自报 hash 与字节数宣称上游官方字节等价。
`minimum_free_vram_gib=28.0` 只是执行前的空闲显存准入下限，不是 peak VRAM 上限或实测值。
它是最终 48 GB 级 `gfx1100` 平台的保守共享默认值，但不证明真实 workload 能放入显存。OpenFold 与
HipFire/OpenMM 的 GPU 分离及单 job 规则不是该门槛的作用。Host 会先获取同一 UID 主机级全局
OpenFold lease，再以 `HIP_VISIBLE_DEVICES` 为 key 获取设备 lease：同设备冲突返回可恢复
`GPU_BUSY`，即使改用另一 GPU 启动第二个 OpenFold job 也返回可恢复 `OPENFOLD_BUSY`。该锁协调
同一用户在不同 workspace 的 ProtBind 进程，但不协调外部 HipFire；HipFire 仍必须显式指定 GPU1，
OpenFold 指定 GPU0。单卡时必须
暂停 GPU LLM/OpenMM；四卡安全默认仍只运行一个 OpenFold job，并把 GPU1–3 保留给其他工具。

每个预测 mmCIF 必须所有坐标有限、精确包含请求的 A（可选 B）蛋白链与单个 Z 配体残基，
且配体重元素计数必须匹配微状态。这不能证明逐原子 identity、键连接/键级、atom mapping 或
立体化学保持；仍需下游化学/PoseBusters 验证。`PROTBIND_TEST_RUNTIME` 只供直接 protocol test，
产品 `WorkerConfig` 明确拒绝它。

Vina worker 直接消费 schema-2 `protbind.selection-bundle`。每个成功候选的 `pose`/`pose_sdf` 都指向
经 Meeko 恢复并通过元素、形式电荷、键序/芳香性、立体化学和 atom-mapping 一致性检查的最佳模式
SDF；原始最佳和全模式 PDBQT 分别保存在 `pose_pdbqt`/`all_modes_pdbqt`，全模式 SDF 保存在
`all_modes_sdf`。验证只把精确引用这些工件且非 fixture 的
`protbind.pose-extraction-receipt` 与 `protbind.receptor-preparation-receipt` 视为 preparation
attestation。

独立实验参考姿态不是筛选提示。它只能在 `DOCKED` 完成后、`VALIDATED` 之前以
`support_reference_pose` 附加，并在生成的 validation batch 中标为 `VALIDATION_ONLY`：

```bash
protbind case attach <run-id> --name reference_pose \
  --file native-ligand.sdf --media-type chemical/x-mdl-sdfile
protbind case resume <run-id> --worker-config configs/protbind-workers.toml
```

有独立参考且 PB-valid、symmetry-aware RMSD ≤2 Å 时，证据级别称
`REDOCKING_RECOVERED`；无参考但两个独立有效方法的 IFP 一致时称 `METHOD_CONSENSUS`。两者都不表示
实验结合支持。只有 docking 证据或制备链未完整 attested 时为 `HYPOTHESIS_ONLY`，硬门失败为
`REJECTED`。

独立的已知位点 redocking 校准可用一个命令运行。原生配体坐标在 Vina 提交姿态前保持
`VALIDATION_ONLY`；对接侧只看到无坐标化学身份和由原生配体边界生成的 box。因为 box 本身来自原生
配体，这仍是 known-site redocking，不是 blind/prospective docking：

```bash
scripts/aiaa-protbind.sh -m protbind_agent benchmark redock \
  --receptor receptor.pdb --native-ligand native-ligand.sdf \
  --receptor-source 'pdb:XXXX/receptor' \
  --native-ligand-source 'pdb:XXXX/ligand' \
  --input-license 'CC0-1.0' \
  --output experiment-results/redock-example
```

冻结十例可用同一个资源受限命令执行或恢复；每个已有 terminal case 只有在 holdout、输入、代码、
工具和配置绑定全部重验后才会复用，且同一时间最多两个单 CPU Vina case：

```bash
scripts/aiaa-protbind.sh -m protbind_agent benchmark redock-holdout-run \
  --holdout holdout.json --holdout-artifacts holdout-artifacts \
  --output experiment-results/fixed10 --max-parallel-cases 2 \
  --vina-bin /path/to/vina \
  --mk-prepare-receptor /path/to/mk_prepare_receptor.py \
  --mk-prepare-ligand /path/to/mk_prepare_ligand.py \
  --mk-export /path/to/mk_export.py
```

启用保守受体修复时必须显式提供不可变 revision，例如
`--conservative-receptor-repair --repair-protected-radius 6.0 --protocol-revision repair-protocol-v1`；
revision 会进入 run plan、batch result 和恢复绑定，不能用另一 revision 复用 terminal artifact。
若 PDBFixer 新增了侧链重原子，可另行启用
`--restrained-sidechain-optimization --sidechain-optimization-iterations 250 1000 5000`。该层固定全部
原始重原子，只移动新增重原子和临时氢，检查非键接距离与手性，移除临时氢后才把第一个通过几何门和
Meeko/RDKit 化学门的结果交给 docking。只有几何未收敛或窄定义的 RDKit valence/sanitize 失败才会
提高 iteration；其他错误仍 fail closed。

2026-07-25 的正式 fixed-ten 结果为 10/10 attempted、8 completed、2 fail closed、0 metric
failures；以 PB-valid 且 symmetry RMSD ≤2 Å 为门，top-1 与 top-5 oracle 均为 7/10。
7XFA 的 PB-valid pose 未恢复，7BTT/7YZU 在原始冻结协议中因缺失侧链而受体制备失败。
事后保守修复只补 6 Å 受保护口袋之外的标准残基重原子，不补 loop、不加 H、不使用
`--allow_bad_res`：它救回 7BTT（0.927 Å），但 7YZU 仍 fail closed，因此不得改写正式成绩。
2026-07-26 又将同一规则冻结为 `repair-protocol-v1` 并重跑全部十例：9 completed、1 fail closed，
独立重算 top-1/top-5 均为 8/10，IFP Jaccard mean/median 为 0.6443/0.6250；7YZU 仍使 gate
incomplete。该 revision 使用的是已经观察过的同一 holdout，属于受控协议修订证据，不是新的前瞻
holdout，也不覆盖原始 7/10 基线。
2026-07-27 再冻结 `repair-protocol-v2-restrained-sidechain`：同一十例 10/10 completed、0 metric
failures，独立 top-1/top-5 均为 9/10，IFP Jaccard mean/median 为 0.6598/0.7014，
`gate_complete=true`。7BTT/7YZU 分别仅移动 13/63 个新增侧链重原子，全部原始重原子最大位移为
0 Å，手性与非键接距离门通过，并在 250 iterations 后通过真实 Meeko。它同样是已观察 holdout 上的
受控协议修订，不能解释为新的 prospective 泛化成绩。
完整哈希、逐例指标和适用边界见
[`refine-logs/EXPERIMENT_RESULTS.md`](refine-logs/EXPERIMENT_RESULTS.md)。Vina score 只作为工具分数，
不得解释为实验结合自由能。

生产科学 worker 默认使用 bubblewrap 独立 network namespace，并只把临时 artifact exchange store
设为可写；当前实现仍将 host `/` 只读绑定给 worker，所以这提供断网与写隔离，但尚不是严格的文件
读取隔离。只应配置经过审查且 hash 固定的 worker。生产导向的 CPU-only Vina 和 validation worker
均已提供；本机已完成 Vina 1.2.7/Meeko 0.7.1、PoseBusters 0.6.5、sPyRMSD 0.9.0 与 ProLIF 2.2.0
的 fixed-ten 回归和 IFP/strain 分析；受约束侧链 v2 已使十例机械回归门完整，但它不是新的前瞻
holdout，盲口袋发现也尚未完成。OpenMM 当前构建只暴露 `Reference`/`CPU`，尚无 HIP platform。
pose extraction/receptor preparation receipt 和自动
quick-Vina selection/validation batch 构建已实现；fixture receipt 仍不能晋升证据等级。真实 OpenFold3 checkpoint inference、
完整 Astex/PoseBench 和 1/2/4×`gfx1100` 调度验证也尚未执行，typed contract 或 fixture 测试不能
当作科学结果。

HIP microbenchmark 的范围仅是三角形匹配内核，不是完整 100k persisted-index/top-512 数据链：

```bash
cmake -S kernels/tripharm_hip -B build/tripharm_hip -DCMAKE_BUILD_TYPE=Release
cmake --build build/tripharm_hip -j
./build/tripharm_hip/tripharm_hip_benchmark \
  --candidates 100000 --molecules 100000 --queries 64 --repetitions 7
```

完整产品决策、科学语义、验收门槛和当前实现差距见
[`DOCs/PROTBIND_PRIVATE_RESEARCH_AGENT_PLAN.md`](DOCs/PROTBIND_PRIVATE_RESEARCH_AGENT_PLAN.md)。

## 结构

```text
src/radeon_agent/
├── agent.py                 # 有最大步数和超时的 Agent 循环
├── backends/                # HipFire / 通用 OpenAI-compatible / Mock
├── tools.py                 # 工具 schema、参数校验、副作用策略
├── memory.py                # 本地 JSONL memory；预留 seekdb/PowerMem 接口
├── hardware.py              # rocminfo / amd-smi / rocm-smi 证据采集
├── benchmark.py             # 预热、重复运行、提示词与请求哈希
└── cross_verify.py          # 可选 gfx1100 ↔ gfx1201 一致性检查
```

核心代码只依赖 Python 标准库。默认测试使用 Mock，不会占用 GPU，也不会碰当前已加载的显存。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

也可不安装，临时使用 `PYTHONPATH=src python -m radeon_agent ...`。

## 暂无 GPU：用 DeepSeek V4 Flash 跑通功能

DeepSeek 只作为 Agent 逻辑和协议的云端 bootstrap，不替代最终的 Radeon/HipFire 本地推理。框架会
安全解析 dotenv，不通过 `source` 执行其中的 shell 内容：

```bash
cp .env.deepseek.example .env
# 在 .env 中填入 DEEPSEEK_API_KEY

PYTHONPATH=src python -m radeon_agent models \
  --backend deepseek --env-file .env \
  --approve-network api.deepseek.com

PYTHONPATH=src python -m radeon_agent chat \
  --backend deepseek --env-file .env --approve-network api.deepseek.com --json \
  "用三条说明本地私有 Agent 的价值"
```

仅发现 `DEEPSEEK_API_KEY` 不会改变本地 `hipfire` 默认后端；每次云端命令都必须同时显式选择
`--backend deepseek` 并批准精确域名。选中后默认模型是 `deepseek-v4-flash`。DeepSeek V4 默认启用
thinking，但 thinking 工具回合要求把
`reasoning_content` 原样带回下一轮；为保证 HipFire、DeepSeek 共用同一套消息协议，当前 DeepSeek
bootstrap 明确设置 `thinking.type=disabled`。后续若需要深度思考，再把 reasoning trace 作为独立的
传输字段完整实现。

可用 `benchmarks/suites/cloud_transport_smoke.jsonl` 验证真实 SSE 和 usage，但任何 `deepseek`
benchmark 都会显示云端警告，并被 cross-verifier 明确拒绝，不能作为 AMD GPU 性能得分证据。

OpenCode 调试也必须显式选择云端启动器；普通 `opencode` 仍只允许本地 HipFire。启动器安全读取
dotenv、强制目标为 `https://api.deepseek.com`、不把 key 写入配置，并临时将
`deepseek-v4-flash` 设为唯一 provider。传给模型的提示、MCP 工具结果及其上下文会离开本机，因此
只能使用公开或脱敏的调试案例：

```bash
scripts/opencode-deepseek.sh \
  --env-file .env \
  --approve-network api.deepseek.com

# 无界面的单轮协议检查；OpenCode 参数必须放在 -- 后
scripts/opencode-deepseek.sh \
  --env-file .env \
  --approve-network api.deepseek.com \
  -- run "只读取当前 ProtBind case 状态并解释下一道 gate"
```

启动器默认关闭 DeepSeek V4 thinking，直到 OpenAI-compatible 工具回合对
`reasoning_content` 的无损续传完成回归验证。它不会放宽 `opencode.json` 的 default-deny 工具权限，
也不会改变阶段 continuation token、单阶段推进和 acceptance receipt 规则。不要把云端运行结果纳入
Radeon 性能证据。

参考：[DeepSeek 首次 API 调用](https://api-docs.deepseek.com/)；
[Anthropic 兼容说明](https://api-docs.deepseek.com/guides/anthropic_api)。

## 启动 HipFire

建议只监听回环地址。HipFire 当前默认监听 `0.0.0.0`，且服务端没有可依赖的 API-key 鉴权；不要把
11435 端口直接暴露到局域网或公网。

```bash
hipfire pull qwen3.5:9b
hipfire config set default_model qwen3.5:9b
hipfire config set host 127.0.0.1
hipfire config set idle_timeout 0
hipfire serve 127.0.0.1:11435 -d
hipfire ps
```

框架只依赖以下稳定契约：`GET /v1/models` 与 `POST /v1/chat/completions`。当前 HipFire 的
`/v1/completions` 并不是本项目依赖项。首次模型加载和 JIT 可能较慢，所以默认 HTTP 超时为 300 秒；
正式基准会先预热，再记录样本。

验证连接并运行 Agent：

```bash
cp .env.example .env
set -a
source .env
set +a

radeon-agent doctor
radeon-agent models
radeon-agent chat "记住我偏好中文，然后总结本地推理的优点" --enable-memory-writes
```

不开 `--enable-memory-writes` 时，模型即便请求 `remember`，权限层也会拒绝写入。本项目没有给模型
任意 shell、文件系统或网络工具。

## W7900 硬件证据

```bash
mkdir -p artifacts
radeon-agent probe > artifacts/w7900-hardware.json
```

输出会保留命令摘要与完整输出的 SHA-256，而不是塞入整份 `rocminfo`。主机应实探到 `gfx1100`。
只要发现 `HSA_OVERRIDE_GFX_VERSION`，正式 benchmark 会直接拒绝运行，避免伪装 GPU 架构。

## 可复现 benchmark

先取得代码 revision、模型权重 SHA-256 和准确量化格式。W7900 上运行：

```bash
radeon-agent benchmark \
  --label w7900-gfx1100 \
  --suite benchmarks/suites/smoke.jsonl \
  --output benchmark-results/w7900.json \
  --repetitions 5 \
  --quantization '<实际量化格式>' \
  --model-revision '<实际模型revision>' \
  --model-sha256 '<实际权重sha256>' \
  --code-revision '<git commit>' \
  --runtime-revision '<hipfire version/commit>' \
  --semantic-config configs/semantic.toml \
  --tuning-config configs/tuning/gfx1100.toml
```

若比赛后取得 `gfx1201`，可使用相同代码、权重、prompt suite、generation 配置和工具 schema 做
可选的非阻塞交叉检查，仅切换机器标签及架构调优记录：

```bash
radeon-agent benchmark \
  --label r9700-gfx1201 \
  --suite benchmarks/suites/smoke.jsonl \
  --output benchmark-results/r9700.json \
  --repetitions 5 \
  --quantization '<与W7900相同>' \
  --model-revision '<与W7900相同>' \
  --model-sha256 '<与W7900相同>' \
  --code-revision '<与W7900相同>' \
  --runtime-revision '<hipfire version/commit>' \
  --semantic-config configs/semantic.toml \
  --tuning-config configs/tuning/gfx1201.toml
```

最后比较：

```bash
radeon-agent compare benchmark-results/w7900.json benchmark-results/r9700.json
```

这个可选检查会核对真实 `gfx1100` / `gfx1201`、suite SHA-256 与 MD5、逐请求 SHA-256、公共语义配置及其
文件哈希、质量检查和 HSA override。生成文本默认不要求逐字一致；跨架构浮点差异可能改变措辞，
正确性应靠确定性字段、数值容差和任务检查判定。可用 `--strict-output` 开启字节级严格模式。

报告中的 `client_ttft_seconds` 是客户端收到第一个非空 `delta.content` 的时间，不等同于 HipFire
内部 prefill/TTFT，也可能包含模型隐藏 reasoning 阶段。后续做内核优化时，应把服务端 profiler 指标与
这个端到端指标分开呈现。

## 测试

```bash
pytest
ruff check .
```

HipFire 是 Kaden Schutt 的独立开源项目，采用 MIT/Apache-2.0 双许可证；本项目仅通过其公开 HTTP
接口调用。参见 [HipFire 仓库](https://github.com/Kaden-Schutt/hipfire) 与
[Serve 文档](https://github.com/Kaden-Schutt/hipfire/blob/master/docs/SERVE.md)。
