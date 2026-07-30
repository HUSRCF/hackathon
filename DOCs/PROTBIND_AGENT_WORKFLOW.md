# ProtBind Agent 端到端工作流

## 1. 工作流结论

ProtBind 采用 **receptor-resolution-first、Vina docking-first、共折叠可选** 的主线：

```text
研究问题与隐私策略
→ 本地输入/化学门禁
→ 用户结构 / 精确缓存 / 授权 RCSB / ESMFold v1 解析受体
→ 三模式药效团查询
→ TriPharm top 512
→ scaffold top 128 + 微状态 + quick Vina
→ top 16；可选 top 8 OpenFold3 复合物预测旁路
→ top 16 evidence-grade Vina（不复用 quick 分数）
→ PoseBusters / RMSD / ProLIF / OpenMM
→ 证据分级、seekdb 入库、带 artifact 引用的本地报告
```

ESMFold v1 只生成受体结构，不接收配体，也不产生蛋白质—配体共折叠结果。缺少 OpenFold3
不会阻止 Vina、验证和报告；系统必须把共折叠证据记为 `unavailable`，而不是把 Vina 或 ESMFold
结果改名为共折叠。

总图见 [ProtBind Agent workflow](../figures/protbind-agent-workflow.md)。

## 2. Agent 与科学工具的职责边界

本地 HipFire/LLM 只负责：

- 把用户意图整理成 `ResearchCase`；
- 选择已注册的受限工具并解释恢复方案；
- 根据确定性 artifact 组织带引用的叙述；
- 检索 seekdb 中已有的文献、失败经验和运行结果。

LLM 不得：

- 生成结构坐标、Vina 分数、ADMET 数字或验证指标；
- 修改工具输出或用自然语言补齐失败阶段；
- 获得任意 shell、任意文件系统或开放网络访问；
- 把 Vina 分数解释为实验结合自由能；
- 把模型姿态解释为真实结合事实。

所有科学数字只能由 RDKit、Gemmi、TriPharm、Vina、PoseBusters、sPyRMSD、ProLIF、OpenMM
或经过门禁的结构模型产生。Agent 只消费带 SHA-256、版本、seed、输入引用和运行时证明的结果。

## 3. 八个执行阶段与九个持久状态

| 阶段 | 主要动作 | 必要产物 | 失败语义 |
|---|---|---|---|
| 1. Case/preflight | 冻结 target、mode、ligand/pocket、seed、privacy；先检查本地 index 和 artifact | `ResearchCase`、输入 QC receipt | 错误身份/不支持化学为 `FAILED` |
| 2. Receptor | 用户结构 → 精确缓存 → 显式授权 RCSB → ESMFold v1 | receptor artifact、resolution receipt | 缺工具/OOM 为 `DEGRADED`；结构硬门失败为 `FAILED` |
| 3. Query | 按三种模式构建 ligand/pocket 药效团；`both` 等权 RRF | 分支 query artifact、分支独立排名 | 查询不足 3 点或来源不明为失败 |
| 4. Screen | 冻结 index，TriPharm CPU/HIP 几何匹配 top 512 | screening bundle、匹配点、overlay、淘汰理由 | 不把几何分称为结合分 |
| 5. Select | scaffold top 128；每分子 ≤4 微状态；≤2 quick Vina；按实测结果选择 top 16 | schema-2 selection bundle、完整 quick-Vina receipt | success 与 failure 的并集精确覆盖请求；失败项无分数/pose/evidence |
| 6. Pose | 进入 `DOCKED` 时先尝试可选 top-8 OpenFold3 旁路，再对 top 16 独立运行 evidence Vina | 规范 docked SDF、原始 PDBQT 与 attestation；cofold pose 可选 | 可选 cofold 失败记旁路状态但不阻塞 Vina 主线 |
| 7. Validate | 化学身份和 PB 硬门；有参考才 RMSD；ProLIF；可参数化时 OpenMM | 每候选 `ValidationBundle` 和 tool evidence | 全部被拒仍可正常出“无合格候选”报告 |
| 8. Report/memory | 证据分级、top 5、引用、seekdb；HipFire 只做受约束叙述 | Markdown、HTML、manifest、seekdb records | 无证据的 claim 必须为 `unknown` |

### 3.1 阶段 1：案例和隐私预检

`ResearchCase` 固定以下内容：

- 1–2 条蛋白链，总长不超过 700 aa；
- `both`、`ligand_only` 或 `pocket_only`；
- 一个普通非共价有机配体，建议不超过 100 个重原子；
- seed、网络许可、精确批准域名和序列上传许可；
- 用户结构、参考配体、口袋残基/中心/box 等输入 artifact。

在任何联网前，先冻结并解析化合物 index、已有结构、配体化学和查询药效团。金属中心、共价配体、
聚合物配体、错误或未指定关键手性、损坏 artifact 和 `HSA_OVERRIDE_GFX_VERSION` 直接 fail closed。

创建 case 前可单独使用 identifier-only public-data 工具获取 RCSB mmCIF、AlphaFold DB mmCIF、
UniProt FASTA、RCSB CCD ideal SDF 或 PubChem CID computed-3D SDF。该工具只构造白名单 URL，
要求用户批准精确域名，并在联网前验证项目内输出路径和固定后缀；它不接受任意 URL、私有序列或
批量查询，也不会把下载物自动接入 run。Gemmi/RDKit 的 parse/QC 摘要和可选 PROPKA 报告属于
acquisition triage。尤其 PROPKA 只提供 pKa/质子化可行性及自身诊断，不能证明缺失 loop/原子已处理、
位点正确或 receptor 已通过科学门禁。

### 3.2 阶段 2：受体解析

受体优先级固定为：

1. 用户显式提供且通过 QC 的结构；
2. 与目标链序列、chain identity 和结构策略精确匹配的本地缓存；
3. 用户显式批准后的 RCSB PDB/UniProt 路径；
4. 本地离线 ESMFold v1 receptor prediction。

RCSB PDB ID 路径只发送 PDB ID；UniProt 路径只发送 accession。只有另行批准
`--approve-sequence-upload` 后才能发送蛋白序列。每个来源必须保留 URL/版本/时间/许可证/原始文件
与抽取 receptor 的独立 SHA-256。

结构 QC 检查：序列身份、1–2 条链、N/CA/C、有限坐标、altloc、金属、声明的共价连接和标准残基。
PDBFixer 后续只允许保守补氢和缺失重原子，不允许静默补长 loop。

ESMFold v1 worker 必须离线、单 GPU、有 GPU lease，并绑定 fair-esm/OpenFold/Torch/权重/代码 hash。
OOM 时只允许 chunk size `128 → 64 → 32`；成功结果仍标为 predicted receptor。当前工作流不会在
`RECEPTOR_READY` 内自动启动 ESMFold：先把 case 停在 `INPUT_VALIDATED`，再在
`RECEPTOR_READY` 之前附加经过校验的 `esmfold_receipt`。直接附加泛化的 `esmfold_structure` 会被拒绝；
receipt 校验目标序列并导入其原始 structure/result metadata。最终 24 aa 单 W7900 warm smoke receipt
记录 8,496,247,808 peak allocated bytes、26.112 s model load、3.653 s inference 和 37.425 s end to end；
它只证明本地断网协议/环境路径，不是性能或准确率证据。

### 3.3 阶段 3：三种研究模式

`both`：

- 从参考配体生成 ligand pharmacophore；
- 从指定口袋生成 complementary pocket pharmacophore；
- 分别筛选，保存两套排名和匹配细节；
- 用等权 reciprocal-rank fusion 合并，不训练黑箱总分。

`ligand_only`：

- ligand pharmacophore 是筛选分支；
- 对受体使用用户/共晶位点，或 fpocket/P2Rank 共识 top 3 生成 docking boxes；
- pocket detection 失败必须显式 `DEGRADED`，不能把全蛋白中心当成默认口袋。

`pocket_only`：

- 从口袋原子环境生成 complementary ligand features；
- 同类型特征 1.5 Å 聚类，最多 12 点、64 个高信息三角形；
- artifact 必须标记 `heuristic=true`。

### 3.4 阶段 4–5：筛选与候选选择

固定漏斗：

```text
100,000 indexed molecules
→ TriPharm top 512
→ Bemis–Murcko unique-scaffold top 128
→ 1–4 microstates per molecule
→ quick Vina on 1–2 microstates per molecule
→ top 16 evidence docking
→ top 8 optional cofold eligibility
```

TriPharm 排序固定为：

```text
query coverage descending
→ median normalized distance error ascending
→ molecule_id ascending
```

选择 bundle 必须绑定完整 screening ranking、library parent identity、scaffold、微状态、receptor、box、
quick-Vina pose 和 tool receipt。selection preparation/finalizer 2.5 还要求先生成 schema/producer 2.0
的 `protbind.docking-box-receipt`，绑定 receptor 完整 `ArtifactRef`/SHA-256、来源、center/size 与
`receptor-cartesian-angstrom` 坐标系；每个维度须为 4–60 Å，体积须 ≤27,000 Å³。门禁从精确 receptor
artifact 重新计算重原子计数/距离，并要求 box 内至少有一个标准蛋白重原子。同一 receipt 必须贯穿
preparation、quick input producer 1.2、每个 request、内外层 evidence、evaluation batch/run metadata 和
最终 selection bundle。

原子重叠只证明 coordinate-frame plausibility，不验证真实生物学位点。`user-center`/`user-residues`
按 `user-hypothesis-only` 解释；`co-crystal-ligand`、`fpocket-p2rank-consensus` 和
`public-benchmark-reference` 必须提供 hash-bound、coordinate-free 的
`protbind.site-derivation-evidence`，绑定 receptor、box/frame、推导方法与 source commitment
SHA-256，且不得把验证参考坐标暴露给筛选路径。即便推导来源通过，也不把位点真实性推断为已证实。
构建器不得只交付 top 8 而丢失前序淘汰理由。主 workflow 现在会确定性生成
`protbind.selection-preparation` 2.5 和最小披露 `protbind.quick-vina-input` 1.2，调用独立的
CPU-only `vina-quick` worker，验证 success/failure 并集精确覆盖全部请求、成功项的
分数/box/seed/evidence 与递归输出闭包，再冻结 top-16
`protbind.selection-bundle`。worker 成功后 batch/完整输出引用 receipt 会在 finalization 前保存，因此
进程中断可恢复且不会重复运行；失败请求没有分数、pose 或 evidence，全失败只产生
`NO_SELECTABLE_CANDIDATES`，不伪造候选。

当 `both` case 显式声明 known-site calibration 时，manual 与 automatic 两条 selection 路径都会在
候选进入 worker 前重新验证 canonical source redock、exact Meeko prepared receptor、preparation
receipt、native-derived box、target ID 及 PB-valid/symmetry-RMSD decision。quick input 只承诺
selection preparation SHA，不接触 native ligand identity/coordinates；receipt 的 PASS 也不等于结合或
亲和力证据。

自动 v1 只接受显式 pocket center/box。fpocket/P2Rank 尚未接入时，ligand-only、残基-only 或缺 box
输入会返回 `SITE_DISCOVERY_UNAVAILABLE`，不会生成 whole-protein box。手工 `selection_batch` attach
仍是兼容入口，不是默认主路径。quick profile 只做 pruning；正式 `DOCKED` 必须独立重跑 Vina。

2026-07-23 的历史 quick profile 1.2/selection config 2.3/input producer 1.1 直接 AIAA v3 quick
smoke 完成 3/3 个真实 Meeko/Vina 请求和完整 36-entry 输出闭包；工具分数为
`-1.942/-1.896/-1.753`，不得解释为实验结合自由能。它使用非已知结合位点的 1CRN、
`chemistry_verified=false` 的 demo index，且仅有 application offline policy，没有经过生产 bubblewrap
边界，因此只支持历史 adapter 的协议/环境可执行性，不支持结合、对接质量、排序、吞吐、OS 隔离或
Radeon 性能结论。历史 evidence 目录为
`experiment-results/aiaa-selection-quick-vina-20260723-v3`，其中 smoke/provenance SHA-256 分别为
`c0e077c2d8c24e59fc4f6d3eece777f1c455b5fd325a7c890152d724339c11ee` 和
`b9b226eb718c2435f7450395f1ac40c2b1ae27a42ce40b02b8a316edfbae1536`；环境 lock/runtime asset 沿用
`f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35`/
`e78b0d4eda4f223e7275270cdde325ae07cd86c490283319b87381853a0a0dd8`，quick/full-Vina code SHA-256 为
`507fd3ac9d311cacd7df516e66d38043f46045d8727e6b36b58b489c8f742be9`/
`e800f8b94c41a343582742d3a8bfbfacaa44a5fbde868f4bfcb58c7deb054334`。site-gate 契约及 runtime
composite hash 已升级，所以 v3/profile-1.2/config-2.3/input-1.1 和 v2 都不能作为当前证据。
2026-07-25 current v4/profile-1.3/config-2.5/input-1.2/box-receipt-2.0 已完成 3/3 请求和 36-entry
closure。smoke/provenance SHA-256 为
`8b19a1d9aa76af0f60dcf0d95c17aa86357598b315d40478dd59c238da0e6f8e`/
`b4f51a5b93e5ac676edb328d7f86b177c009c7e90a11b8a9e444d0f0e0c7739f`；quick/full code SHA-256 为
`ed48d5b6884bd4a572cd7315d046b557b2bfd0ea3d1953dd72ee5acfa7458d14`/
`a181906553cc0960a9e51f02856df3b2bc527fd4a0aad24321d61bd92760f56c`。1CRN box 有蛋白原子重叠但
仍无 biological-site derivation，index 仍为 unverified fixture，且仅 application-offline direct
adapter；因此不支持 docking-quality、binding、ranking、OS-isolation 或 Radeon-performance 结论。

使用 `chemistry_verified=true` mini-index 的历史 v3 production workflow 已把 receipt
`641d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a` 绑定到 preparation 和 quick
input，并到达 `SELECTED` worker launch；但宿主 bubblewrap 在配置 loopback/network namespace 时因
`RTM_NEWADDR: Operation not permitted` 被拒绝。运行按设计停在 `DEGRADED`，最后完成 `SCREENED`，记录
recoverable `WORKER_CRASH` 且无 quick 结果。其 production manifest 位于
`experiment-results/aiaa-selection-quick-vina-20260723-v3/production-workspace/runs/production-isolation-smoke-v3/manifest.json`，
SHA-256 为 `730e54551f807ed18896257fa3d2f47ba9da3a930c8bc2b7b8a2c6ebff585746`。`protbind doctor` 的
`runtime_details.worker_network_isolation.status` 当前实测为 `present_but_unusable`（probe rc=1）；合法
状态为 `missing | present_but_unusable | usable`，只有 `usable` 能通过生产 OS 隔离门。application
offline 环境变量不等于 OS 隔离。该平台阻断不能通过关闭 `isolate_network` 或 fixture bypass 绕过；
应在具备所需 namespace capability 的宿主上重跑。

### 3.5 阶段 6：docking-first 姿态路径

恢复进入 `DOCKED` 时，workflow 首先处理可选 cofold。若配置了 OpenFold worker，且在冻结
`SELECTED` 后、`DOCKED` 前附加了 batch/checkpoint/environment lock，host 会以
`stage=COFOLDED`、`previous.stage=SELECTED` 的 schema-2 envelope 运行 top-8 旁路；状态为
`RUNNING → COMPLETED`，或写入 `UNAVAILABLE`/`FAILED_RECOVERABLE`。随后正式 Vina 的 envelope 仍以
`SELECTED` 为 `previous`，只在旁路成功时额外携带 `cofold_evidence_bundle`，所以 cofold 不是主状态或
必需依赖。

`support_openfold_batch` 必须是与 frozen selection、screening、index、receptor 和 quick-Vina 身份
一致的 `protbind.cofold-input-batch`。当前仓库只有严格验证器，没有从自动 selection 生成该 batch 的
公开 builder/CLI；因此现阶段须由受信外部工具构造并审计后 attach，不能描述成自动 OpenFold 流程。

必选主路径：Vina worker 随后直接消费 schema-2 selection candidates，对 top 16 运行 CPU
Meeko/AutoDock Vina，保存 prepared receptor/ligand、box、seed、所有 modes、最佳 pose、工具分数及
语义。规范姿态字段 `pose`/`pose_sdf` 是通过 Meeko 恢复的最佳模式 SDF；原始最佳与全模式 PDBQT
分别保存在 `pose_pdbqt`/`all_modes_pdbqt`，全模式 SDF 保存在 `all_modes_sdf`。候选准备或 docking
失败是该候选的结构化失败，不代表其他候选失败。

每个成功候选还必须带 `protbind.pose-extraction-receipt`，证明元素/同位素、形式电荷、键序/芳香性、
立体化学、氢数和 atom mapping 在 PDBQT→SDF 恢复中保持；bundle 级
`protbind.receptor-preparation-receipt` 绑定原始 receptor、规范化 PDB 输入和 prepared receptor
PDBQT。验证阶段只接受精确引用这些 artifact、全部检查通过且 `test_fixture=false` 的 attestation。

可选旁路：只有本地 checkpoint、官方 runtime、`low_mem`、ROCm Triton、单 sample、显存 admission
和离线门禁全部通过时，才对 top 8 运行 OpenFold3。其置信度是模型置信度，不是结合概率。

当前统一输出为 schema-2 `protbind.docking-bundle`：

```text
candidate identity
receptor identity
required Vina pose/evidence
optional cofold pose/evidence when the side task completed
pose/receptor preparation receipts and structured candidate failures
```

reference pose 不写入 docking bundle；它只在 DOCKED 后作为 `VALIDATION_ONLY` support 进入自动生成的
validation batch。这使验证阶段不依赖某个共折叠器，也避免把 docking-only continuation 伪装成
`COFOLDED`。

### 3.6 阶段 7：多证据验证

顺序固定：

1. 配体原子、元素、键序、手性和 parent mapping；
2. PoseBusters chemical/geometric hard gate；
3. 只有在 `DOCKED` 后、`VALIDATED` 前显式附加 `support_reference_pose` 时才做 symmetry-aware RMSD；
4. ProLIF 计算氢键、疏水、π、盐桥及 IFP；
5. 只有普通、成功参数化的非共价体系才做 OpenMM 局部最小化几何门。

独立 redock regression 的 ProLIF 路径不会让 RDKit 对整条远端受体推断拓扑。它先对两份配体做
receipted hydrogen handling，再取“距 docked 或 reference 任一重原子 8 Å 内”的完整残基并集；
receipt 必须证明所有保留原子身份不变、坐标最大偏差不超过 0.002 Å。该 crop 只用于 IFP，不回流到
docking 或 RMSD。生产 validation worker 尚需接入同一 helper，当前不能把 regression adapter 的
能力写成所有在线 validation 已自动具备。

redock benchmark 另提供 opt-in 保守受体修复：只补标准残基且所有缺失重原子都在原生配体 6 Å
保护半径之外时才允许执行；缺失 loop/整残基、口袋内缺原子、修复后化学几何仍非法都会 fail
closed。修复不加 H、不移动原有重原子、不使用 `--allow_bad_res`，Meeko 仍是唯一质子化 authority。

`repair-protocol-v2-restrained-sidechain` 在上述重原子修复与 Meeko 之间增加一个 opt-in 受约束几何
层。OpenMM 只让新增侧链重原子和为力场临时增加的 H 移动，全部原始重原子以 zero-mass 固定；输出前
移除全部 H，并验证原始原子 identity/坐标（≤0.002 Å）、新增原子键长、涉及新增原子的非键接 vdW
距离比（≥0.60）以及 CA/ILE-CB/THR-CB 手性符号。任何 identity、参数化、距离或手性门失败均显式
fail closed。默认 minimization iteration 上限为 `250 → 1000 → 5000`；只有受约束几何未收敛或
Meeko/RDKit 报告 valence/sanitize 化学错误时才试下一档，不能把工具 crash 或 unsupported template
伪装为“多试几次”。第一个通过全部几何门和真实 Meeko 检查的结构成为 exact docking input，并以
attempt receipt 绑定。OpenMM 能量只作 preparation diagnostic，不是结合能、自由能或稳定性结论。

证据分级：

- `REDOCKING_RECOVERED`：独立参考姿态标为 `VALIDATION_ONLY`，且 PB-valid、symmetry RMSD ≤2 Å；
- `METHOD_CONSENSUS`：无参考真值，但两个独立有效方法的 IFP 达到门限；
- `HYPOTHESIS_ONLY`：PB-valid，但只有 docking 或证据准备链尚未完全 attested；
- `REJECTED`：身份、碰撞、应变、结构或参数化硬门失败。

`REDOCKING_RECOVERED` 只说明某个已知复合物的姿态恢复门通过，`METHOD_CONSENSUS` 只说明方法间
一致；两者都不能写成实验结合支持。OpenFold3 不可用时，无参考候选通常最多为
`HYPOTHESIS_ONLY`。若所有候选均被拒绝，应正常生成“没有通过门禁的候选”报告，而不是把 run 标成
失败。

### 3.7 阶段 8：报告、RAG 和记忆

先生成确定性 Markdown/HTML 报告，再允许 HipFire 将其组织为自然语言。每个结论必须引用文献页码/
章节或 artifact ID。报告必须展示：

- 每个候选为什么进入、为什么淘汰；
- 工具版本、seed、模型/权重/代码/硬件 hash；
- 缺失工具、unsupported reason 和失败记录；
- Vina/cofold/实验事实之间的语义界线；
- top 5 之外的漏斗计数和未独立验证候选数量。

seekdb 是案例、作业、证据、文档 chunk 和 artifact 引用的唯一精确状态源。PowerMem 只存偏好、常用
协议和失败经验摘要，并且每条记忆必须回指 seekdb job/artifact。

## 4. schema-2 持久状态与辅助任务

当前 schema-2 主状态固定为：

```text
CREATED → INPUT_VALIDATED → RECEPTOR_READY → INDEXED
→ SCREENED → SELECTED → DOCKED → VALIDATED → REPORTED
```

schema-1 manifest 保留原有 `COFOLDED` 线性状态，但只能读取，不能 resume 或重写。schema 2 把
`COFOLDED` 保留为 worker task 名，并用 manifest 的 `cofold_status`、`cofold_record` 和
`cofold_failure` 记录可选候选级证据：

```text
NOT_REQUESTED | UNAVAILABLE | RUNNING | COMPLETED | FAILED_RECOVERABLE
```

主状态只表示必需路径已完成；辅助任务的输入/配置 hash、artifact 和失败仍写入 manifest。恢复时：

- 已完成主阶段和辅助任务不重复计算；
- 可恢复失败保留最后成功主状态；
- 修复 capability 后从缺失任务继续；
- 输入、代码、模型或硬件身份改变时拒绝复用旧 cache。

### 4.1 Agent 阶段闭环

交互式 Agent 使用 `StageGateController`，而不是让 LLM 直接请求从当前状态跑到 `REPORTED`：

```text
case_status
→ PREFLIGHT gate + content-addressed receipt
→ 用户/Agent 解释检查项和所需处理
→ case_advance(fresh continuation token)
→ 正好一个主阶段
→ POSTFLIGHT acceptance + content-addressed receipt
→ next PREFLIGHT gate
```

continuation token 绑定 run、完整 manifest SHA-256、下一阶段和控制策略 SHA-256。附件、阶段完成或
任何 manifest 变化都会使旧 token 失效。preflight 和 postflight 都调用核心 workflow 的同一深度
审计定义；MCP/UI 不得自行缩减验收规则。支持以下控制决策：

- `READY`：允许用本次 token 推进一次；
- `ACCEPTED`：本次阶段、输出和绑定验收通过；
- `NEEDS_ACTION`：缺少明确输入/能力，必须处理后重新 gate；
- `RETRYABLE`：有可恢复 failure，但只能显式重试；
- `UNSUPPORTED`/`FAILED`：停止，不得静默降级；
- `COMPLETE`：全部主阶段已有 accepted record。

本地 MCP 提供 doctor、受约束 public-data fetch、case create/status/advance/attach/report/dossier/
pose-view、artifact metadata 和 control history；不提供 shell、任意文件内容、原始坐标或开放
网络。public-data fetch 是唯一网络工具，只接受白名单 source、公开 identifier、精确批准域名和
项目内目的路径，不会 attach 或推进阶段。`case_dossier` 区分
stage record 与 postflight acceptance，`case_pose_view` 只返回无坐标的 artifact/验证/box/几何
QA 摘要；两者都不能修改 manifest 或推进阶段。所有路径必须位于 project root，network-enabled case
也会被 MCP create 拒绝。OpenCode 项目配置只启用本地 HipFire 与该 MCP，mutating tools 均为
`ask`，其他工具默认 `deny`。MCP 1.14 的 AIAA stdio 已完成真实 initialize/list-tools 握手；OpenCode
可执行文件和 HipFire 对话尚未在当前宿主实跑，因此这只是协议/权限闭环证据，不是完整交互 demo。

## 5. 失败和降级矩阵

| 情况 | 处理 |
|---|---|
| 金属中心、共价/聚合物配体、超过边界 | `FAILED` 或明确 unsupported；不静默降级 |
| RCSB 未授权或无精确命中 | 不联网；尝试 ESMFold v1 |
| ESMFold v1 OOM | 128/64/32 重试；仍失败则 `DEGRADED` |
| fpocket/P2Rank 均不可用且无用户 box | `DEGRADED` at site discovery |
| 单个分子无法标准化/参数化/对接 | candidate failure；继续其他分子 |
| OpenFold3 未配置 | 保持 cofold `NOT_REQUESTED`；继续 Vina 主线 |
| OpenFold3 无 batch/checkpoint 或当前不可用 | 写 cofold `UNAVAILABLE`；继续 Vina 主线 |
| OpenFold3 OOM/crash | 写 cofold `FAILED_RECOVERABLE`；继续 Vina 主线 |
| PoseBusters-invalid | candidate `REJECTED` |
| OpenMM 不可参数化 | 明确 unsupported；不得伪造能量/稳定性 |
| 所有候选被拒绝 | 正常 `REPORTED`，结论为无通过门禁候选 |
| worker crash/缺依赖 | `DEGRADED`，保留断点和 artifact |
| 宿主不允许 bubblewrap network namespace/loopback | `DEGRADED`，保留最后完成阶段并记录 recoverable `WORKER_CRASH`；迁移到合格宿主，不得关闭生产隔离 |
| hash/provenance/输入身份不一致 | `FAILED`，禁止复用 |

## 6. 双 gfx1100 调度

- GPU0 是互斥 scientific lane：TriPharm HIP、ESMFold v1、可选 OpenFold3、OpenMM 依次运行；
- GPU1 留给 HipFire 和交互界面；不得让 OpenFold 抢占两张卡；
- Vina、Meeko、PoseBusters、ProLIF、sPyRMSD 和 seekdb 默认 CPU；
- 一个任务只看到一个规范的 `HIP_VISIBLE_DEVICES` index；禁止其他 mask 和架构伪装；
- 1/2/4 张卡验收矩阵仍是计划项；当前折叠 adapter 每 job 只支持一张卡，尚未实现或验证多 GPU
  adapter 调度。上游多 device 模式也只分发独立任务，不合并显存。

ESMFold v1 的 24 aa 本机 smoke 峰值约 7.91 GiB，但这不是 700 aa 上限输入的显存保证；调度仍以
实际 admission、OOM retry 和 receipt 为准。

## 7. 当前实现与下一步

| 能力 | 当前状态 |
|---|---|
| Case/schema、隐私门、artifact/manifest、恢复审计 | 已实现并有回归测试 |
| 用户结构/本地缓存/RCSB 拦截 | 已实现；公开 1CRN 网络与离线缓存 smoke 通过 |
| ESMFold v1 | AIAA + 单 W7900 24 aa 离线真实推理通过；最终 receipt 为 26.112 s load、3.653 s inference、37.425 s end-to-end；须在 RECEPTOR_READY 前 attach receipt，自动调度待接 |
| ESMFold2 / AlphaFold DB | 无 ESMFold2 runnable worker；AlphaFold DB accession-only 候选获取已实现，但尚未自动接入 receptor resolver |
| TriPharm CPU、三模式/RRF | 已实现；三模式 protocol smoke 到 `SCREENED` |
| persisted-index TriPharm HIP top 512 | kernel microbenchmark 已通过；完整 backend 待接 |
| schema-2 main state / optional cofold task | 已实现；schema 1 只读，DOCKED 不再依赖 cofold |
| scaffold/microstate/quick-Vina selection | config 2.5/profile 1.3/input producer 1.2/box receipt 2.0、原子重叠 coordinate-frame plausibility 门、独立 site-derivation evidence、known-site calibration consumer、全谱系、自动 CPU-only worker、精确 success/failure 覆盖、完整输出闭包、缓存恢复和 top-16 finalizer 已实现；v4 direct smoke 为 3/3、36-entry closure，但仅 application-offline/unverified-chemistry；历史 verified-chemistry production run 在 bubblewrap namespace 建立处 fail closed，生产隔离仍待重跑；fpocket/P2Rank 待接 |
| evidence-grade Vina | worker 可直接消费 SELECTED；规范 SDF、原始 PDBQT 和逐候选失败均已绑定 |
| OpenFold3 | 官方 ROCm runtime validator 通过；checkpoint/真实 inference 未运行；仅有 cofold batch 验证器，无公开 batch builder/CLI，保持专家导入的可选旁路 |
| validation | docking-derived batch/toolchain、pose/receptor attestation 已实现；原始 fixed-ten 为 8 completed/2 failed、top-1/top-5 7/10；`repair-protocol-v1` 为 9 completed/1 failed、独立 8/10；`repair-protocol-v2-restrained-sidechain` 为 10/10 completed、独立 top-1/top-5 9/10、IFP mean/median 0.6598/0.7014、`gate_complete=true`；三版结果分别保留，v2 仍是已观察 holdout 上的受控修订 |
| deterministic report | 最终 Markdown/HTML 已实现；任意 checkpoint 的 JSON/Markdown/HTML run dossier 已接，区分 computed 与 accepted |
| seekdb/BGE-M3 | adapter 和本地权重门已实现；完整资料集需显式导入 |
| HipFire/OpenCode/PowerMem | 受限 stdio MCP、逐阶段 gate/acceptance receipt、OpenCode default-deny 配置和 project skill 已实现，MCP 1.14 握手通过；OpenCode executable/HipFire TUI 实跑与 PowerMem 仍待完成 |
| 六页 Web UI | 已接真实 run dossier、pose 列表、loopback receptor/ligand view、box/pocket styles 和浏览器 PNG；3Dmol.js 2.5.4 有固定 URL/SHA-256/许可证安装器，仍待公开真实复合物浏览器验收 |

下一组实现应按以下顺序进行：

1. 在 `protbind doctor` 报告 OS network isolation 为 `usable` 的宿主上，以同一冻结的
   chemistry-verified mini-index 重跑生产自动 selection；通过隔离门后，再用预先冻结且带独立位点
   来源证据的 public receptor/box 跑通同一复合物三种模式的 Vina-only 端到端报告；box receipt
   2.0 的原子重叠只能作为坐标系 plausibility，独立的 coordinate-free derivation evidence 也只证明
   来源推导，不得把两者写成生物学位点验证；
2. 保留原始 fixed-ten、`repair-protocol-v1` 与已完成的
   `repair-protocol-v2-restrained-sidechain`；用新的、真正 result-blind 外部集检验 v2 泛化，不得
   把同一已观察 holdout 的 9/10 写成 prospective 成绩；
3. 将 fpocket/P2Rank 共识、已实现的保守受体修复/ProLIF crop 和 ESMFold v1 receipt 自动调度接入
   主流程；
4. 用一个公开、可再分发的 canonical DOCKED/VALIDATED run 做浏览器端 3Dmol 与 PNG 人工验收，
   并保存 viewer 版本/hash 与 camera/selection 参数；截图仍不得进入科学 evidence grade；
5. 若继续 OpenFold3，先实现由 frozen selection 生成 cofold batch 的公开 builder/CLI，再决定是否
   导入 checkpoint 作为附加方法一致性证据。
