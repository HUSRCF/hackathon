# ProtBind 本地私有蛋白质—配体科研 Agent

端到端运行语义与 receptor-resolution-first、docking-first 状态机见
[`PROTBIND_AGENT_WORKFLOW.md`](PROTBIND_AGENT_WORKFLOW.md)。该文档明确把 OpenFold3 共折叠改为
可选候选级证据，避免其缺失阻塞 Vina、验证和报告主线。

版本：0.2（schema-2 实施基线）  
日期：2026-07-25  
目标：AMD AI DevMaster Track 2

## 1. 产品结论与科学边界

ProtBind 定位为在单张 AMD Radeon GPU 上运行的本地私有药物发现科研 Agent。目标链路是：

```text
靶点与研究假设
→ 结构/化学质量控制
→ 10 万化合物 TriPharm-HIP 药效团筛选
→ 候选多样性与快速对接
→ 入围候选 Vina docking；可选蛋白—配体共折叠作为旁路证据
→ PoseBusters/ProLIF/OpenMM 多证据验证
→ seekdb + PowerMem 持久记忆
→ HipFire 本地 LLM 生成带引用的科研报告
```

支持模式：

- `both`：配体和口袋假设分别筛选，以等权 RRF 融合；报告保留两个分支的独立排名和几何分。
- `ligand_only`：用参考配体药效团筛选；候选口袋由用户输入或 fpocket/P2Rank 共识产生。
- `pocket_only`：从残基、中心/box 或预计算口袋药效团产生互补特征。

v1 只接受 1–2 条蛋白链、总长不超过 700 aa，以及一个普通非共价有机配体（建议不超过 100 个
重原子）。金属中心、共价/聚合物配体、错误或不明确的立体化学和无法参数化的体系必须显式拒绝或
跳过对应验证。任何结构、打分、RMSD、相互作用或 ADMET 数字必须来自具名工具 artifact。LLM 只可
规划、解释和总结。

语义硬约束：

- Vina 数值只能称为 pose-ranking tool score，不得称为实验结合自由能。
- 共折叠输出只能称为模型预测姿态，不得称为真实结合事实。
- TriPharm 数值是几何匹配分，不是结合分或活性预测。
- 缺失能力或失败结果不得由 LLM、fixture 或经验常数填补。

## 2. 模型与运行时决策

| 组件 | 优先级 | 决策 |
|---|---:|---|
| ESMFold v1 | P0 | 仅在无实验/缓存结构时从私有序列产生受体；官方 fair-esm 独立环境，chunk 128→64→32 OOM 重试 |
| OpenFold3 0.4.3 | P0 可选 | 首选共折叠引擎；官方源码运行于继承 AIAA Torch/Triton 的独立 overlay，MSA-free/direct-CIF，生产 host 强制 `low_mem` |
| ESMFold2 | P1 future-only | 当前无可运行 worker；只有通过最终 `gfx1100` 平台的 3-complex 门禁后才可评估为共折叠替代/第二意见 |
| ColabFold | P2 | 只保留本地 A3M/AF3 JSON `MSAProvider` 接口；公共 MSA Server 默认禁止上传序列 |
| RCSB | P0 导入 | 已实现显式联网授权、精确链/序列门和来源 receipt |
| AlphaFold DB | P1 候选获取已实现 | 已有 accession-only、精确域名授权、curl + receipt 的独立候选获取；尚未自动接入 receptor resolver |

依据与上游：

- [ESMFold v1](https://github.com/facebookresearch/esm) 与
  [模型卡](https://huggingface.co/facebook/esmfold_v1)。
- [OpenFold3](https://github.com/aqlaboratory/openfold-3)、
  [安装](https://openfold-3.readthedocs.io/en/latest/Installation.html) 和
  [推理](https://openfold-3.readthedocs.io/en/stable/inference.html)。
- [ESMFold2](https://huggingface.co/biohub/ESMFold2) 与
  [ROCm attention NaN 记录](https://github.com/Biohub/esm/issues/322)。
- [ColabFold](https://github.com/sokrypton/ColabFold) 与
  [ROCm JAX](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/native_linux/install-jax.html)。

ESMFold v1 实际离线依赖三个文件：ESMFold checkpoint、ESM2 3B backbone 及 contact-regression
checkpoint。三者按 role/size/SHA-256 计算一个复合权重 identity；worker 继续使用 PyTorch
`weights_only` restricted unpickler，并只允许审查过的 argparse/OmegaConf globals，不能为兼容旧
fair-esm 而全局改成不安全 pickle。最终单张 W7900 post-hardening 24 aa warm 烟测 receipt 记录峰值
allocated VRAM 8,496,247,808 bytes、model load 26.112 s、inference 3.653 s、end-to-end 37.425 s；
该数字只证明断网协议与本地 runtime 可运行，不是性能 benchmark。现有导入权重集虽已 hash-pinned，
但未证明与上游当前下载对象逐字节
相同，因此不写成官方 release 等价或模型准确率证据。

结构与姿态预测按能力路由：用户结构/本地精确序列缓存/获授权 RCSB 命中时不再折叠受体；否则旧
ESMFold v1 仅生成受体。对 top-8 姿态候选，当前唯一实现的可选 adapter 是仍待真实 checkpoint 门禁的
OpenFold3；ESMFold2 只是未来候选，并未接入。复合物预测器不可用时，流程必须把共折叠证据标为
unavailable，继续受体 + Vina + PoseBusters/ProLIF 主线；不得把 ESMFold v1 受体或 Vina pose 伪称为
共折叠结果。

ESMFold2 的晋级门禁为：同一 3 个公开复合物（至少一个任意 SMILES）、在单张 `gfx1100` 上完全断网、
无 NaN、原子/键序/手性/映射一致、`low_mem` 峰值不超过 40 GiB、3/3 PoseBusters 可解析，且在双卡
主机上不占用 HipFire 的 GPU1；PB-valid 数和参考 RMSD 中位数不劣于 OpenFold3。未全部通过时只保留
通用 `ComplexPredictor` 接口。

## 3. 代码架构

```text
src/radeon_agent/                 通用 Agent、HipFire、硬件与 cross-verifier
src/protbind_agent/
├── models.py                     领域 schema 与 v1 硬门
├── artifacts.py                  SHA-256 内容寻址存储
├── manifest.py                   状态机、cache key、失败/恢复记录
├── chemistry.py                  可选 RDKit 标准化、ETKDGv3、六类特征
├── structure.py                  Gemmi 检查和启发式口袋药效团
├── preparation.py                保守 PDBFixer helper（不补长 loop）
├── tripharm.py                   SQLite index 与 CPU reference
├── fusion.py                     branch-preserving RRF
├── selection.py                  scaffold、微状态与真实 quick-Vina selection receipt
├── cofold_batch.py               可选 top-8 OpenFold input receipt 硬门
├── workflow.py                   确定性阶段编排
├── worker_protocol.py/sdk.py     隔离环境 JSON 协议
├── validation.py                 证据等级和术语
├── validation_input.py           从 DOCKED bundle 构建并绑定 validation 输入/toolchain
├── redocking.py                  独立 redocking box、受体/配体准备和 symmetry RMSD helper
├── redock_benchmark.py            sealed-reference Meeko/Vina/PoseBusters/sPyRMSD public calibration
├── knowledge.py                  pyseekdb + 本地 BGE-M3 gate
├── capabilities.py               doctor/capability 证据
├── web.py                        FastAPI 六页面
└── cli.py                        protbind CLI
workers/                          ESMFold v1/OpenFold3/Vina/验证适配器与隔离边界；ESMFold2 未实现
kernels/tripharm_hip/             HIP triangle matcher 和 microbenchmark
```

Core 目标环境为 Python 3.12。RDKit/Gemmi/PDBFixer、Meeko/Vina、PoseBusters、ProLIF、sPyRMSD、
OpenMM、pyseekdb 与 PowerMem 都是显式 capability；缺包不能触发隐式算法替代。OpenFold3 使用独立
进程与约 36 MiB 的 AIAA-backed overlay，并继承 AIAA 已验证的 ROCm Torch/Triton；ESMFold v1
作为独立子进程复用 AIAA/core overlay 并绑定完整运行时 hash；ESMFold2 尚无 runnable worker，未来
若晋级也必须使用不污染主环境的独立实验环境。
HipFire 只绑定 `127.0.0.1`。DeepSeek V4 Flash 只可验证传输/协议，云端
数字不能进入 Radeon 性能证据。

worker 请求/响应必须含：schema 版本、job ID、输入/输出 `ArtifactRef`、参数、seed、模型 revision、
权重/代码 SHA-256、计时、峰值显存、warning 和结构化 error。Host 使用 argv + `subprocess`，不使用
shell，不默认转发 API key 或完整环境。每个科学阶段接收一个内容寻址的 `protbind.stage-input`
依赖包，其中包含 case、library/input、query/support artifact，以及上游的全部科学输出和独立 receipt；
恢复时重新构造并比对该依赖包，避免把“上一步第一个文件”误当作完整输入。worker 环境禁止
`HSA_OVERRIDE_GFX_VERSION` 和疑似 credential 变量，其非敏感值只以 canonical hash 进入 config identity。

生产 pipeline 默认对 worker 启用 bubblewrap network namespace。Host 递归复制当前 stage
显式声明的 artifact graph 到临时 exchange store，只允许 worker 写该 store，输出 hash 验证后再导入
主 store。当前 bwrap 命令仍将 host `/` 以只读方式 bind；因此已有 OS 级断网和写隔离，但不能
宣称已严格隔离对其他 host 文件的读取。生产配置只允许 hash-pinned/审查过的 worker；
非隔离执行只在 `fixture-only` 测试中有显式 bypass。

### 3.1 OpenFold3 可选 adapter contract

`workers/openfold3_worker.py` 已实现具体的断网 CLI adapter，固定为 OpenFold3 0.4.3 commit
`0bb17be5199846e806b6347b6e17c6249c88ff1b`，请求的 `model_revision` 必须精确为
`openfold3-0.4.3@0bb17be5199846e806b6347b6e17c6249c88ff1b`。schema 2 中 `COFOLDED` 只是
可选 side-task 名，不是主状态。配置该 side task 时，其 envelope 必须含：

- `support_openfold_batch`：与本 run screening artifact、library index 和允许 receptor 绑定的
  `protbind.cofold-input-batch` v1.0；
- `support_openfold_checkpoint`：本地 checkpoint，其 artifact SHA-256 必须等于 `weight_sha256`；
- `support_openfold_environment_lock`：AIAA 基座、overlay、ROCm validator 与官方源码 allowlist 的
  `aiaa-openfold3-environment.json` 审计 artifact。

自动路径必须先以 `--stop-after selected` 冻结 selection，再在 `DOCKED` 前附加上述三项。cofold 在
进入 `DOCKED` 时先执行；其 envelope 为 schema 2、`stage=COFOLDED`、
`previous.stage=SELECTED`。随后正式 Vina 的 envelope 仍以 `SELECTED` 为主上游，并只在 cofold 成功时
额外带入 `cofold_evidence_bundle`。当前仓库只有 `protbind.cofold-input-batch` 的严格验证器，没有从
自动 selection 生成它的公开 builder/CLI；所以 OpenFold 仍是由受信外部工具构建并审计 batch 后的
专家导入路径，不能写成全自动主链。

`code_sha256` 不是单一 Python 文件 hash。OpenFold allowlist 从 Python 实际 import 的唯一
`openfold3` package root 递归扫描，并验证 `openfold3.run_openfold` 也位于该 root；它不依赖
distribution `RECORD`/file list，所以适用于当前官方 editable AIAA-backed overlay。谓词精确选取：

- `openfold3/` 下所有以 `.py` 结尾的文件；
- 相对路径中位于 `core/data/resources/` 下的所有资源；
- 以 `model_setting_presets.yml` 结尾的 preset。

Adapter 对 allowlist 中的相对路径/SHA-256 pair 排序后计算 runtime-manifest SHA-256。同时对全部
`src/protbind_agent/**/*.py` 计算 canonical repository-relative path/SHA-256 manifest，因为 adapter 会导入
这些 host 模块。最终以 canonical JSON 将 schema version、adapter SHA-256、
`protbind_runtime_sha256`、environment-lock SHA-256、installed OpenFold runtime-manifest SHA-256 和上述
revision 一起绑定为复合 `code_sha256`。运行时还必须精确满足：

```text
distribution version = 0.4.3
entry point = run_openfold = openfold3.run_openfold:cli
scm tag = 0.4.3
scm distance = 0
scm node = g0bb17be5199846e806b6347b6e17c6249c88ff1b
scm dirty = false
allowlisted file count = 317
runtime manifest SHA-256 = 742e9bf654b13f67783d095a2327af3ed31163580eaa7b4c548e8a8eb2e68010
```

`PROTBIND_TEST_RUNTIME=1` 只能在直接 worker protocol test 中放宽官方 allowlist/ROCm 门禁，且输出
显式标为 fixture runtime。产品 `WorkerConfig` 对该保留变量直接拒绝，TOML 不得含它。

Batch 验证器硬性检查 receptor 与链序列、完整 screening ranking 及 library-index parent identity、
确定性且 scaffold 唯一的 Bemis–Murcko top-128、每个保留分子 1–4 微状态，以及每个分子
1–2 个 quick-Vina 评估。每个 Vina receipt 必须把分数、receptor、pose artifact、box center 和 box
size 与同一分子/微状态绑定。排序固定为 Vina 分数升序 → `molecule_id` →
`microstate_id`；top-16 和每分子最优微状态的 top-8 必须精确重现该排序。Adapter 仅消费
该 receipt，不自己伪造多样性、微状态或快速对接。

Adapter 生成官方 query JSON/runner YAML。Query JSON 顶层写入 `seeds: [request.seed]`，每个
query 都显式设置 `use_msas=false`、`use_paired_msas=false` 和 `use_main_msas=false`；runner YAML
重复 seed 并关闭 MSA server。Adapter 只用当前隔离环境的解释器调用
`python -m openfold3.run_openfold predict --use-msa-server=False`，不允许用参数替换可执行路径；只从
本地 artifact 提供可选 direct-CIF template。输出的
`protbind.cofold-bundle` 绑定候选/微状态、CIF、confidence 和运行 metadata；confidence 只称
模型置信度，runtime attestation 和 OpenFold 有效配置 artifact 一并保留。子 CLI 未暴露峰值显存时，
adapter 明确返回 unavailable，不估算。

每个模型 CIF 还必须通过 Gemmi 结构门：坐标全部有限；蛋白链必须精确为 A（以及双链时的 B）
并与请求序列一致；配体必须是 Z 链的一个非空残基；不允许其他非水分子链；配体重元素
计数必须等于请求微状态。当前该门仍不能验证逐原子 identity、键连接/键级、atom mapping 或立体
化学保持；重元素计数相等不得被表述为键/手性保持，仍需下游化学与 PoseBusters 验证。

产品 worker argv 必须从绝对路径调用 `scripts/aiaa-openfold3.sh`，由它进入显式的 AIAA-backed
overlay；不能由 ambient working directory 选择 Python/pixi project，也不能在该 overlay 再安装一份
Torch。`support_openfold_environment_lock` 必须冻结 AIAA 基座审计、overlay lock、官方源码 commit
与源码 allowlist hash。
资源策略保守固定为：每个 OpenFold job 的 `HIP_VISIBLE_DEVICES` 只能是一个规范数字 GPU index，
并拒绝同时配置 `ROCR_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES` 或 `GPU_DEVICE_ORDINAL`，
runner 只使用一个 trainer device、强制 FP32 (`precision: 32-true`)，且调度层一次只允许一个
OpenFold job。上游的 `devices: 2/4` 模式也只把独立 query 分发到不同 GPU、不汇聚 VRAM；当前
ProtBind adapter 不暴露该多 device 模式。双 `gfx1100` 主机默认将 GPU0 用作科学计算 lane（OpenFold
结束后再运行 OpenMM），GPU1 留给 HipFire。`checkpoint_name`
固定为 OpenFold3 0.4.3 文档中的 `openfold3-p2-155k`。产品 worker 必须核对其官方字节数
2,287,928,196 bytes，并同时核对 `weight_sha256`。当前没有已确认的官方小/中/大模型系列；
checkpoint 后缀和文件大小也都不是显存档位或峰值显存估计。基线只用
hash-attested checkpoint 与 `low_mem`。当前仓库尚未冻结独立可信的上游 checkpoint SHA-256 allowlist；
本地 `weight_sha256` 只绑定 run identity，自报 hash 加字节数不能证明与上游对象逐字节等价。
`minimum_free_vram_gib` 共享默认为 28.0，仅是运行前读取保留 GPU 空闲显存的 admission
floor，不是 peak VRAM cap、目标或实测值。该值是最终 48 GB 级 `gfx1100` 平台的保守共享默认值，
但不证明真实 query 能放入显存；adapter 仍硬拒绝低于 24 GiB。Host 会先为同一 UID 非阻塞获取
主机级全局 OpenFold lease，再以 `HIP_VISIBLE_DEVICES` 为 key 获取设备 lease；同设备冲突
返回可恢复 `GPU_BUSY`，另一设备上的第二个 OpenFold job 返回可恢复 `OPENFOLD_BUSY`。该 lease
协调同一用户跨 workspace 的 ProtBind 进程；HipFire 不参与此锁，所以双 `gfx1100` 时仍必须显式
将 HipFire 指定到 GPU1，而 OpenFold 指定 GPU0。单卡必须时间片串行化 GPU LLM/OpenMM；四卡的
安全默认仍只运行一个 OpenFold job，并把 GPU1–3 保留给其他工具。生产 host 还强制
`engine=openfold3`/`official-openfold3`、官方 bundle producer，并把 query/runner/run metadata、
checkpoint、environment lock、stage envelope、provenance 与完整 raw-output inventory 交叉绑定；
缺失/额外 artifact 或 engine alias 均 fail closed。当前官方源码已在 AIAA Torch
2.12.1+rocm7.2/Triton 3.7.1 上通过 ROCm/Evoformer kernel validator，并使用确定性 fake CLI
完成 inference protocol contract test；尚未运行真实 checkpoint、
1/2/4×`gfx1100` 调度/性能、3 复合物
bake-off 或 PoseBusters 科学验证。

## 4. 领域接口与状态机

公开类型：

- `ResearchCase`、`ResearchMode`、`PrivacyPolicy`；
- `ArtifactRef`（不含内部路径）；
- `MoleculeRecord`、`ScreenHit`、`PoseCandidate`；
- `ValidationBundle`、`EvidenceClaim`、`EvidenceGrade`；
- `RunManifest`、`StageRecord`、`FailureRecord`。

正常状态顺序固定为：

```text
CREATED → INPUT_VALIDATED → RECEPTOR_READY → INDEXED
→ SCREENED → SELECTED → DOCKED → VALIDATED → REPORTED
```

schema-1 manifest 的旧 `... → COFOLDED → DOCKED ...` 状态只允许读取，不能 resume 或重写。
schema 2 将可选共折叠单独存为 `cofold_status`、`cofold_record`、`cofold_failure`，状态为
`NOT_REQUESTED | UNAVAILABLE | RUNNING | COMPLETED | FAILED_RECOVERABLE`；其失败不会回滚或阻塞
Vina 主线。`DEGRADED` 保存主路径最后成功阶段与可恢复失败，`FAILED` 保存不可恢复的输入/科学门禁
失败。每个阶段 cache key 为 schema、stage、输入 hash 和 config hash 的 SHA-256。恢复前重新验证全部
artifact 的大小、内容 hash 和 record cache key，已完成 artifact 不重写。

CLI：

```text
protbind doctor
protbind index build|inspect
protbind case run --mode both|ligand_only|pocket_only
protbind case run --vina-environment-lock <aiaa-lock.json>
protbind case run --rcsb-pdb-id <id>|--rcsb-uniprot-accession <accession> \
  --approve-network <exact-domain> [--approve-sequence-upload]
protbind case resume|show|report
protbind case attach --name esmfold_receipt  # after INPUT_VALIDATED, before RECEPTOR_READY
protbind case attach --name selection_batch  # 兼容的外部 batch 导入口
# after SELECTED, before cofold/DOCKED:
protbind case attach --name openfold_batch|openfold_checkpoint|openfold_environment_lock
protbind case attach --name reference_pose   # only after DOCKED, before VALIDATED
protbind knowledge import
protbind knowledge fetch --approve-network <domain>
protbind ask
protbind serve
protbind benchmark [--index ... --query ...]
protbind benchmark redock --receptor <pdb> --native-ligand <sdf> --output <directory> \
  --receptor-source <public-id> --native-ligand-source <public-id> --input-license <license>
protbind benchmark redock-holdout-run --holdout <holdout.json> \
  --holdout-artifacts <store> --output <directory> --max-parallel-cases 1|2
```

生产前 `protbind doctor` 必须报告
`runtime_details.worker_network_isolation.status=usable`。该字段的闭集是
`missing | present_but_unusable | usable`；当前宿主实测 `present_but_unusable`、bubblewrap probe
return code 1。application offline 环境变量与 `offline_default` 不是 OS 网络隔离，不能替代该门。

ESMFold 的 generic structure attach 被拒绝；`esmfold_receipt` 必须与目标序列一致，host 才会导入其
原始 structure/result metadata 并注册精确序列缓存。OpenFold attach 目前也不等于自动 batch 构建；
公开 builder/CLI 仍待实现。

## 5. 科学工具链

### 5.1 输入与口袋

- Gemmi 解析 PDB/mmCIF，并检查链数、残基数、altloc、金属和非标准残基。
- 蛋白结构在折叠前按“用户 artifact → 本地精确序列缓存 → 显式授权 RCSB → 折叠”解析。RCSB
  显式 PDB ID 只访问官方 [File Download Service](https://www.rcsb.org/docs/programmatic-access/file-download-services)
  的 `files.rcsb.org`；UniProt 发现访问官方 [Search API](https://search.rcsb.org/) 的
  `search.rcsb.org` 后再下载坐标；无标识的
  sequence search 还必须单独批准序列上传。任何网络调用前先冻结并解析 index、查询药效团、配体
  化学和已有本地 artifact；损坏的精确缓存 fail closed。候选必须具有唯一或显式的 chain 分配、
  坐标序列精确匹配、完整 N/CA/C、有限坐标、无未解析 altloc，且不含金属或声明的共价连接。显式
  chain ID 进入缓存身份，不能由同序列的另一 chain 命中。mmCIF 检查 `_struct_conn`；PDB 检查 LINK
  和蛋白—配体 CONECT，未声明只记为 partial evidence。默认明确
  下载 deposited asymmetric unit；biological assembly 只在用户给出 assembly ID 时下载，并使用
  独立 URL/缓存身份。HTTP transport 禁用环境代理，避免精确域名批准被代理静默改道。receipt 保存
  URL、UTC 获取时间、archive revision/header、CC0、原始下载 mmCIF 与链抽取 receptor 的独立
  artifact SHA-256、选择/丢弃的 chain ID 与 QC；raw artifact 同时绑定进 run input；导入坐标按 RCSB 官方
  [CC0 usage policy](https://www.rcsb.org/pages/usage-policy) 记录。Search 原始 JSON、request body/hash
  和完整返回候选列表目前均未冻结；receipt 只保留请求类别、是否上传序列与实际尝试的候选。歧义/错序列/金属候选不静默
  降级为有效结构。
- PDBFixer 只允许保守补氢/缺失重原子/altloc；不得静默补长 loop。
- RDKit 使用 MolStandardize、保持原始与标准 parent、ETKDGv3 单线程固定 seed，每分子最多 4 构象。
- 位点优先级：用户 box/残基/共晶配体 > fpocket/P2Rank 共识 top-3 > 明确失败。
- 口袋启发式把面向空腔的蛋白 donor/acceptor、芳香、疏水和带电环境映射为互补配体特征；同类型
  1.5 Å 聚类，最多 12 点、64 个高信息三角形。artifact 必须带 `heuristic=true`。

上述 PDBFixer 制约和 fpocket/P2Rank 共识是产品验收要求。当前仓库已有保守 PDBFixer helper、
P2Rank 受控 CLI/CSV hypothesis adapter、私有库序列 QC 和 accession-only UniProt 本地比对，但
尚未把 fpocket/P2Rank 共识、自动 box 构造和这些库 entry 的 case materialization 接入三种端到端
模式，不应宣称该验收项已完成。

### 5.2 TriPharm

特征类型固定为：

```text
Donor, Acceptor, Aromatic, Hydrophobe, Positive, Negative
```

输入支持带预计算特征的 JSONL；安装 RDKit 后支持 SDF、SMILES、CSV 和 Parquet（Parquet 另需
PyArrow）。原始分子不覆盖，索引只保留标准 parent，微状态推迟到 top-128。默认 bin 0.5 Å、容差
1.0 Å，全部进入 index metadata 和 manifest config hash。

CPU reference 对每个查询三角形和候选三角形枚举六种对应，验证类型与三条边误差，保存匹配原子、
候选构象和 Horn/Kabsch 4×4 overlay。排序固定为：

```text
query coverage ↓ → median normalized distance error ↑ → molecule_id ↑
```

`both` 模式使用等权 RRF，保留每个分支的 rank、几何分和具体 match。HIP 当前实现是可编译、可在
gfx1100 实跑的 triangle-match microbenchmark；它正确处理六种对应，并按 molecule 聚合 64-bit query
mask 和每查询最小误差。它尚不是 persisted-index/top-512 全链，不可冒充最终 100k benchmark。

### 5.3 漏斗与证据

目标漏斗：

```text
100,000 warm index
→ TriPharm-HIP top 512
→ Bemis–Murcko scaffold diversity top 128
→ 每分子最多 4 微状态、最多 2 个快速 Vina
→ top 16
→ 可选 top 8 OpenFold3（不可用时显式跳过；ESMFold2 为未来项）
→ evidence-grade Vina + 物理/相互作用验证
→ top 5 报告
```

当前 host 已实现 schema-2 `protbind.selection-preparation` 与 `protbind.selection-bundle` builder：
确定性 Bemis–Murcko top-128、每分子最多 4 个微状态、最多 2 个 quick-Vina 请求，以及只有在真实
quick-Vina success/failure 并集精确覆盖计划请求、且成功项 pose/evidence 完整后才能冻结的 top 16。
失败项不得含分数、pose 或 evidence。主 workflow 已接入独立 CPU-only
`vina-quick` worker：使用稳定 request ID、最小披露输入、精确请求覆盖、递归 artifact 输出闭包、完整
receipt 和可恢复缓存；当前版本为 selection preparation/finalizer 2.5、quick profile 1.3、quick input
producer 1.2、docking-box receipt 2.0。receipt 绑定 receptor 完整 `ArtifactRef`/SHA-256、来源、
center/size 和 `receptor-cartesian-angstrom`；box 各维必须为 4–60 Å，体积必须 ≤27,000 Å³。它从精确
receptor artifact 重新计算原子重叠，并要求 box 内至少有一个标准蛋白重原子。该 receipt 必须原样
贯穿 preparation、quick input、每个 request、内外层 evidence、evaluation batch/run metadata 和最终
selection bundle。

原子重叠只证明 coordinate-frame plausibility，不证明生物学位点。`user-center`/`user-residues` 保持
`user-hypothesis-only`；`co-crystal-ligand`、`fpocket-p2rank-consensus` 和
`public-benchmark-reference` 必须提供 hash-bound、coordinate-free 的
`protbind.site-derivation-evidence`，绑定 receptor、box/frame、推导方法与 source commitment
SHA-256，并证明验证参考坐标未暴露给筛选路径。该独立证据只支持“推导来源已绑定”，不支持“位点
真实性已证实”。quick 分数只用于 pruning，正式 `DOCKED` 仍独立重跑 Vina。自动 v1 只接受
显式 pocket center/box；fpocket/P2Rank 未接入时返回 `SITE_DISCOVERY_UNAVAILABLE`，不得猜 whole-protein
box。可选
`protbind.cofold-input-batch` 继续为 top-8 OpenFold side task 提供严格身份门，但目前只有验证器、没有
公开 builder/CLI。

selection 2.5 已把 target-specific known-site calibration 接入 `both` 的 manual/automatic selection
gate：receipt 必须回指 canonical source redock，并精确绑定 Meeko prepared receptor、preparation
receipt、native-derived box、target ID 与 PB-valid/symmetry-RMSD decision。1IEP real consumer smoke
的 calibration/preparation/quick-input SHA-256 分别为
`650aff4a6910ec549f1b807f80d8c3ef424be9461818c5b5886cd6bfbffa4d82`/
`489c594be042ff257046ada66dffbd69fb9659ace14589308a9a916f1a281d17`/
`26836f7ca9832907924469d0fdc66d6edc753f2d55b1c31a0bd86c2fe15c1d1b`，且 quick input 不含 native
reference。该门只校准 known-site pose recovery，不支持 binding/affinity claim。

2026-07-23 的历史 profile 1.2/selection config 2.3/input producer 1.1 直接 AIAA v3 quick smoke 完成
3/3 个真实 Meeko/Vina 请求和完整 36-entry 输出闭包，工具分数为
`-1.942/-1.896/-1.753`（不是实验结合自由能）；它使用非已知结合位点的 1CRN、
`chemistry_verified=false` 的 demo index，并只验证 application offline policy，没有经过生产
bubblewrap。该结果只证明历史 adapter 的协议/环境路径，不支持结合、对接质量、排序、吞吐、OS 隔离
或 Radeon 性能结论。历史 evidence 目录为 `experiment-results/aiaa-selection-quick-vina-20260723-v3`；
smoke/provenance SHA-256 分别为
`c0e077c2d8c24e59fc4f6d3eece777f1c455b5fd325a7c890152d724339c11ee`/
`b9b226eb718c2435f7450395f1ac40c2b1ae27a42ce40b02b8a316edfbae1536`。environment lock/runtime asset
保持 `f1081dd9ffd8097e488a1a2ac2d12ee946efb1a6a22582c4d306f546c2d79f35`/
`e78b0d4eda4f223e7275270cdde325ae07cd86c490283319b87381853a0a0dd8`，quick/full-Vina code SHA-256 为
`507fd3ac9d311cacd7df516e66d38043f46045d8727e6b36b58b489c8f742be9`/
`e800f8b94c41a343582742d3a8bfbfacaa44a5fbde868f4bfcb58c7deb054334`。site-gate 契约与 runtime
composite hash 已升级，因此 v3/profile-1.2/config-2.3/input-1.1 和 v2 都不能作为当前契约证据。
2026-07-25 v4/profile-1.3/config-2.5/input-1.2/box-receipt-2.0 已完成 3/3 请求与 36-entry closure；
smoke/provenance SHA-256 为 `8b19a1d9aa76af0f60dcf0d95c17aa86357598b315d40478dd59c238da0e6f8e`/
`b4f51a5b93e5ac676edb328d7f86b177c009c7e90a11b8a9e444d0f0e0c7739f`。它仍是 user-center、
unverified-chemistry、application-offline protocol smoke，不是 docking science 或 production evidence。

独立 redock regression 1.1 先在六例 retrospective pilot 保留完整失败分母；随后冻结了
PoseBusters/PoseBench 308-case source list 上 result-blind 的 fixed-ten holdout。正式批次 10/10
attempted、8 completed、2 fail closed、0 metric failures；PB-valid + symmetry-RMSD≤2 Å 的 top-1
和 top-5 oracle 均为 7/10。holdout/run-plan/batch/regression-v2 文件 SHA-256 分别为
`01d9fd57f31ef006601b6a1e982d2cf020d50761bc9c8f5bfe61497ccc064ca3`、
`a3eb1736d63fb76086c111fddb56fff75793d307c76cc4f409a2e7fe266ff99c`、
`89deee7806792f120ef23b6cf72aab8bd910c54ac6b929fbfa5b2dbd1a36ad06`、
`cab37219c7918a852a35b0296199da8cc60e5bdb52c584f72a8d517e232a85a8`。两个 receptor-preparation
failure 使 `gate_complete=false`；失败仍留在十分母，不能以八个完成例为分母。

fixed-ten regression v2 的 ProLIF 只读取 receipted ligand H preparation 和距两份配体重原子 8 Å
内的完整残基并集；保留原子 identity 与坐标不变，避免远端受体 proximity bonding 污染局部 IFP。
八例 IFP Jaccard mean/median 为 0.6248/0.6125。保守 PDBFixer heavy-atom-only remediation 只允许修复
原生配体 6 Å 保护区外的标准残基缺失重原子，不重建 loop、不加 H、不调用 `--allow_bad_res`。
回溯性消融中 7BTT 以 0.927 Å top-1 被救回，7YZU 仍在受体制备 fail closed；该消融不能改写
正式 7/10。

2026-07-26 将该规则显式冻结为 `repair-protocol-v1`，并以相同 holdout、输入、seed、工具和最多两个
单 CPU Vina case 重跑全部十例；run plan 同时绑定 revision 与 35-file source manifest
`0fcb6c2b...`。结果为 10 terminal、9 completed、1 fail closed，独立 real-tool PB/sPyRMSD/ProLIF
重算的 top-1/top-5 均为 8/10，IFP Jaccard mean/median 为 0.6443/0.6250。7BTT 修复 13 个口袋外
重原子后恢复，7YZU 修复 63 个口袋外重原子后仍被 Meeko 拒绝，因此 `gate_complete=false`。这是对
已观察 holdout 的受控协议 revision，不是新的 prospective holdout；原始 7/10 结果保持不变。

2026-07-27 冻结 `repair-protocol-v2-restrained-sidechain`。它在 conservative heavy-atom repair 与
Meeko 之间固定全部原始重原子，只让新增侧链重原子和临时 H 接受 OpenMM ff14SB cutoff 几何优化；
输出在移除临时 H 后硬检原始原子 identity/≤0.002 Å 位移、新增原子键长、非键接 vdW 距离比≥0.60
及 CA/ILE-CB/THR-CB 手性。iteration schedule 为 `250→1000→5000`，只有几何未收敛或窄定义的
RDKit/Meeko valence/sanitize 失败可进下一档，其他错误 fail closed。相同十例正式结果为 10/10
completed、0 metric failures，独立 top-1/top-5 均为 9/10，IFP Jaccard mean/median
0.6598/0.7014，`gate_complete=true`。7BTT/7YZU 分别只移动 13/63 个新增重原子，两者 250
iterations 即通过真实 Meeko。source/run-plan/batch/independent-regression SHA-256 分别为
`56cc89f42e8397340b64546bc1824adbe80730eab0b5ae1a632fbf2d1c718e5d`、
`c6d6b4117b8db4503ad535530195a70b0f439847c6e6bba86a3c0b21634c813c`、
`43e413f46bea1396ee653f250c0e789b1022821354807ff14ecf91d14bec75a1`、
`90e558fc93657b15dd5bf46862aa19e6d20e36cccef62d1520aa38ab6c9c8de1`。v2 仍是同一已观察 holdout
上的受控修订，不替代原始 result-blind 7/10，也不是 prospective 泛化估计。OpenMM receipt 的能量
只称 preparation diagnostic，不能解释为结合能或稳定性。

另一个 `chemistry_verified=true` 的历史 v3 production workflow 已将 docking-box receipt
`641d7aa6fbab3eba685e954989ea0de1d51bbda52875f308d242154b87c3747a` 绑定到 preparation 和 quick
input，并到达 `SELECTED` worker launch；宿主 bubblewrap 在配置 loopback/network namespace 时返回
`RTM_NEWADDR: Operation not permitted`。run 保持 `DEGRADED`、最后完成 `SCREENED`，并记录 recoverable
`WORKER_CRASH`，没有 quick 结果。其 manifest 位于
`experiment-results/aiaa-selection-quick-vina-20260723-v3/production-workspace/runs/production-isolation-smoke-v3/manifest.json`，
SHA-256 为 `730e54551f807ed18896257fa3d2f47ba9da3a930c8bc2b7b8a2c6ebff585746`。该宿主能力阻断不能通过
关闭 `isolate_network` 或 fixture bypass 绕过，生产隔离门仍未通过。214 tests 是历史 v3 contract
的回归计数，不代表当前 site-gate 契约已经完成重跑。

生产导向的离线 `vina_worker.py` 已能直接消费 `SELECTED` bundle，并实现 CPU-only Meeko/Vina 调用、
精确运行时哈希、分数门禁及候选失败显式化。每个成功候选以 Meeko 恢复的最佳模式 SDF 作为规范
`pose`/`pose_sdf`，另存最佳/全模式 PDBQT 和全模式 SDF；逐候选
`protbind.pose-extraction-receipt` 检查元素/同位素、形式电荷、键序/芳香性、立体化学、氢数与 atom
mapping，bundle 级 `protbind.receptor-preparation-receipt` 绑定 receptor PDB→PDBQT 重原子/残基身份。

`validation_input.py` 从精确 `DOCKED` bundle 自动生成 validation batch，并对本地
PoseBusters（必需）及可用的 sPyRMSD/ProLIF/OpenMM 生成 hash-bound toolchain/provenance。
`validation_worker.py` 已实现强制 PoseBusters dock/redock 门、可选 sPyRMSD/ProLIF/OpenMM、逐工具
runtime/evidence artifact 与显式 unsupported 原因。只有精确引用上述非 fixture receipts 且所有检查
为真时，`preparation_attested` 才能为真；fixture 仍不能晋级证据等级。独立
`benchmark redock` 还会冻结 Python、全量 ProtBind source manifest、工具版本/二进制、配置、来源、
许可证和 composite run identity，并在 Vina 提交姿态前隔离原生参考坐标。2026-07-23 共尝试六个
公开输入：三个完成 known-site redocking，三个分别因未指定关键双键手性、缺失受体侧链重原子和保留
sulfate 而 fail closed。在三个完成例中，以 `PB-valid AND symmetry RMSD ≤2 Å` 为门，top-1 为 2/3、
reference-aware top-5 oracle 为 3/3；这不是六输入的 3/3 成功率，也不是盲对接、虚拟筛选命中率、
亲和力或完整十例回归。

参考 RMSD 只接受在 `DOCKED` 之后、`VALIDATED` 之前显式附加的 `support_reference_pose`；生成的
validation batch 将其标为 `VALIDATION_ONLY`，普通 ligand hypothesis structure 不等于已对齐实验参考
姿态。OpenMM 当前只称“局部最小化几何门”，不得把粒子数一致称为参数化证明，也不得把最小化称为
稳定性模拟。

最终不训练不透明总分。PoseBusters-invalid、化学身份、碰撞或参数化硬门失败先淘汰，然后分级：

- `REDOCKING_RECOVERED`：独立参考姿态仅作验证，PB-valid 且 symmetry RMSD ≤ 2 Å；
- `METHOD_CONSENSUS`：无参考真值，但两个独立有效方法的姿态及 IFP 一致；
- `HYPOTHESIS_ONLY`：仍是合理假设，但缺少独立一致证据；
- `REJECTED`：化学、碰撞、参数化或结构门禁失败。

前两级分别表示 redocking 姿态恢复门和方法间一致，不是“实验支持结合”的别名。

## 6. RAG、记忆和隐私

[pyseekdb](https://github.com/oceanbase/pyseekdb) 是 seekdb 的 Python SDK。生产配置以 embedded seekdb
作为案例、分子、任务、证据、文档 chunk、版本和 artifact 引用的唯一精确状态源；使用结构化、全文和
向量混合检索。当前 adapter 使用 collection `upsert`、`refresh_index` 与 `hybrid_search`，未安装
pyseekdb 时明确失败，不用 SQLite/JSONL 冒充生产 seekdb。

[PowerMem](https://github.com/oceanbase/powermem) 只保存偏好、失败经验、常用协议和历史工作流摘要；每条
记忆必须回指 seekdb job/artifact，不替代原始结果。BGE-M3 必须从显式本地目录加载；缺权重或
FlagEmbedding 时不自动联网下载。

联网默认禁止。`knowledge fetch` 只接受 HTTPS、精确批准的域名、大小上限，并禁止自动跟随重定向；
URL query 不进入 artifact source。RCSB sequence search 和 ColabFold 都需独立的
`--approve-sequence-upload`，普通联网许可不能代替序列上传许可；PDB ID/UniProt 路径不发送序列。
Agent 永远没有通用 shell、任意文件系统或开放网络工具。`.env`、API key、私有
序列和绝对内部路径不得进入日志、报告、benchmark 或公开提交。

Web UI 使用 FastAPI 和 HTMX-compatible fragment，不引入 Node 构建链。Cases、筛选漏斗、3D 姿态、
证据、RAG 和 Radeon 性能为六个本地页面，另有逐 run dossier。3Dmol.js 2.5.4 通过固定 URL、
SHA-256 和完整许可证的显式安装器进入 workspace；运行时只从本地已验证 asset 提供，缺失或篡改时
fail closed，禁止 CDN fallback。受体/配体坐标只发给 loopback 浏览器，不进入 MCP 响应。

## 7. 验收

功能：

- 同一公开复合物跑通三种模式；
- 预取权重/资料后断网从输入到 HTML/Markdown；
- 任意阶段中断恢复且已完成 artifact hash 不变；
- 每个候选可解释进入/淘汰原因与证据；
- 金属、共价、错误手性、缺关键残基、OpenMM 参数化失败、OOM/worker crash 均有显式状态。

科学：Astex/PoseBusters 10 复合物小回归，主指标 `PB-valid AND symmetry-RMSD ≤2 Å`，同时报告
top-1/top-5、IFP recovery、碰撞与应变。100k ChEMBL 固定 hash 子集只用于规模/性能，不能包装成
命中率。更完整评测使用 [PoseBench](https://github.com/BioinfoMachineLearning/PoseBench) 1.1+ 和
[PoseBusters](https://github.com/maabuu/posebusters)。

当前科学基线是 result-blind fixed-ten known-site redocking：10/10 attempted、8 completed、2 fail
closed，PB-valid + symmetry RMSD≤2 Å 的 top-1/top-5 均为 7/10；八个完成例均有独立 PB/RMSD、
IFP 和 strain/clash 指标。7XFA 的 PB-valid pose 未恢复，7BTT/7YZU 原始协议在受体制备失败。
另行冻结的 `repair-protocol-v1` 全十例重跑为 9 completed/1 failed、独立 top-1/top-5 8/10；7YZU
仍在受体制备 fail closed。`repair-protocol-v2-restrained-sidechain` 为 10/10 completed，独立
top-1/top-5 9/10、IFP mean/median 0.6598/0.7014，并使该修订的机械 gate 完整。
完整结果见 [`../refine-logs/EXPERIMENT_RESULTS.md`](../refine-logs/EXPERIMENT_RESULTS.md)。这仍不是
blind-pocket、虚拟筛选命中率、结合或亲和力证据；v2 用的是已观察的同一 holdout，必须另用新的
result-blind 外部集验证泛化。

Radeon：

- CPU/HIP top-512 集合完全一致；浮点模式 recall ≥ 0.999；
- 100k warm-index HIP ≥ CPU 5×，报告 p50/p95、VRAM、transfer/kernel；
- OpenFold3 的无 Triton/非 `low_mem` 对照只允许进入 standalone benchmark，不得进入 schema-2
  生产 cofold side-task evidence；生产只记录官方 ROCm Triton + `low_mem`；
- OpenFold3 每 job 只暴露一个 GPU 且不并发；双 `gfx1100` 时 GPU1 留给 HipFire，OpenMM HIP 在
  OpenFold 退出后复用 GPU0；
- OpenMM 同时报 CPU/HIP、最小化后能量和几何差；
- HipFire 单列 TTFT、吞吐和工具成功率；
- 1/2/4×`gfx1100` 未来验收使用相同代码、权重、输入、seed/config hash；当前折叠 adapter 只支持
  单 GPU/job，尚无多 GPU adapter 调度证据。始终禁止 `HSA_OVERRIDE_GFX_VERSION`；`gfx1201` 仅作
  可选、非提交门禁的交叉检查。

2026-07-21 的本机 kernel microbenchmark 验证（非完整筛选）：gfx1100，100,000 candidate triangles、
100,000 molecule IDs、64 query triangles、seed 20260721；CPU/HIP mask exact、recall 1.0、float-bit
mismatch 0、分配 28,401,280 bytes VRAM。一次 7-repetition 运行的 CPU 为 0.050214102 s、kernel p50
为 0.000063760 s、p95 为 0.000069685 s，H2D 为 0.047288104 s、D2H 为 0.002991605 s；该数字只证明内核与
协议可行，不能外推为完整索引筛选吞吐，也不能据 kernel-only 时间宣称端到端达到 5×。

## 8. 排期、当前完成度和停止条件

| 阶段 | 目标 | 当前状态（2026-07-30） |
|---|---|---|
| 7/21–22 | schema/interface、OpenFold3/ESMFold2 bake-off | schema-2 Vina-first 状态机和可选 cofold task、协议及 pinned OpenFold3 adapter contract 完成；AIAA-backed 官方 ROCm validator 已通过；无公开 cofold batch builder，真实 checkpoint/三复合物 bake-off 未运行，ESMFold2 future-only |
| 7/23–25 | 输入、ESMFold v1、index、CPU | 输入、optional RDKit/Gemmi、index/CPU 完成；ESMFold v1 adapter 已以 hash-pinned 三权重集在单张 W7900 完成 24 aa 断网 warm 烟测（8,496,247,808 bytes allocated VRAM；26.112 s load、3.653 s inference、37.425 s end-to-end），目前是 receipt attach 而非自动调度，且尚非 benchmark、准确率或官方字节等价验证 |
| 7/26–28 | HIP、top-k、gfx1100 profile | triangle matcher 已实跑；persisted-index/top-k 未接 |
| 7/29–31 | Vina/PB/ProLIF/OpenMM | Vina canonical-SDF/PDBQT、pose/receptor receipts、selection 2.5/quick 1.3/input 1.2/box receipt 2.0、site evidence、known-site calibration consumer、validation lineage 及 v4 direct smoke 已实现；原始 fixed-ten 为 7/10，v1 为 8/10，受约束侧链 v2 为 10/10 completed、独立 9/10 且 revision gate complete；独立 local-crop ProLIF/strain/clash 完成，新的 prospective 外部集、production isolation 和 OpenMM HIP 仍未完成 |
| 8/1–2 | seekdb/PowerMem/BGE-M3/HipFire | seekdb+BGE gate、default-deny OpenCode research/library skills、受限 MCP 与逐阶段闭环 gate/acceptance receipt 完成；独立蛋白/配体 CAS catalog、scan/apply/move 收据、UniProt identity states 和完整 CLI 已接；DrutAI 因环境/许可证/模型 provenance/bake-off 门禁保持禁用；OpenCode/HipFire TUI 实跑与 PowerMem 报告器未接 |
| 8/3 | Web、3D、报告、demo | 最终报告 + checkpoint dossier、真实 pose scene、无坐标 Agent QA、loopback 3Dmol/PNG 与固定 asset 安装器已接；公开真实复合物浏览器验收和完整 demo 未完成 |
| 8/4–6 | 1/2/4×gfx1100 调度、回归、许可、复现、提交；gfx1201 可选 | 未执行 |

明确排除：现有私有 PLC/Pocket-GVP、ciffixer、PLK 代码/checkpoint；非商业 ESM3 权重；GNINA、smina、
CUDA MMseqs2 或云 DeepSeek 主链；分子生成、合成路线、FEP/RBFE、临床结论；PDBbind/完整 ChEMBL/
第三方权重再分发。ColabFold 940 GB 本地数据库、长期双引擎维护、ADMET-AI 和大规模 PoseBench 属于 P2。

停止条件很简单：如果工具未安装、输入不支持、worker 未配置、artifact hash 不一致或科学门禁失败，
ProtBind 必须以 `DEGRADED`/`FAILED` 结束并给出原因；不得为了演示完整流程而伪造任何阶段。
