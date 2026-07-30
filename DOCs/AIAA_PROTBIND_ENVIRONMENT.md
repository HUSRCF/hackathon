# ProtBind AIAA 比赛环境

## 结论

比赛主环境采用共享基座加轻量覆盖层：AIAA 提供经过验证的 Python 3.12、ROCm PyTorch、RDKit、
Gemmi、OpenMM、PDBFixer、FastAPI 和旧 ESMFold；仓库内的
`.venv-aiaa-protbind` 只补充 Vina/Meeko、PoseBusters、ProLIF、sPyRMSD、seekdb、embedding
adapter 与 Dimorphite-DL 微状态枚举。这样不会复制约 37 GiB 的 AIAA，也不会修改其 ROCm 栈。

OpenFold3 使用另一个约 36 MiB 的 `.venv-aiaa-openfold3`，但继承 AIAA 的
Torch `2.12.1+rocm7.2` 和 Triton `3.7.1`。官方 0.4.3 ROCm 验证器（包括 Evoformer
Triton kernel）已通过。上游 pixi.lock 仍作为版本参考，但比赛主机不再重复安装约
数 GiB 的 ROCm Torch wheel。

旧 ESMFold v1 使用约 20 MiB 的 `.venv-aiaa-esmfold-v1` 和约 92 MiB 的固定版本
legacy OpenFold 源码/CPU attention 扩展，同样继承 AIAA 的 ROCm Torch、fair-esm
`2.0.0` 与 OmegaConf。覆盖层不含 Torch。fair-esm 2.0.0 的两个可变 dataclass 默认值
不兼容 Python 3.12；运行时只在官方源文件 SHA-256 完全匹配时把它们改为
`default_factory`，不修改张量运算或权重。

此前出现 Torch 下载不是 AIAA 不兼容，而是上游 pixi 环境按设计完全隔离，无法继承 Conda/AIAA
site-packages，因此会严格按自己的 lock 再解析一份 Torch。比赛环境现已停止该安装路径；overlay
requirements 明确不包含 `torch`、`triton` 或 ROCm 包，并由官方 validator 直接检查 AIAA 提供的版本。

## 创建和使用

首次联网安装：

```bash
scripts/bootstrap-aiaa-protbind.sh --download-vina
scripts/bootstrap-aiaa-openfold3.sh --clone
scripts/bootstrap-aiaa-esmfold-v1.sh --clone
```

随后所有 ProtBind Python 命令都通过同一个入口运行：

```bash
scripts/aiaa-protbind.sh -m protbind_agent doctor
scripts/aiaa-protbind.sh -m pytest
scripts/aiaa-protbind.sh -m protbind_agent index build \
  --input examples/library.features.jsonl \
  --output experiment-results/demo-index.json
```

启动器默认设置 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，并禁止
FlagEmbedding 自动导入 TensorFlow。确实需要预取公开模型时，只能在明确授权的独立
导入步骤中临时关闭离线变量；私有序列和内部路径不得发送。

本机 `pdftotext`/`pdftoppm` 来自 Poppler，可用于有文字层 PDF；当前没有 Tesseract/OCRmyPDF，
所以扫描页会在 extraction receipt 中明确标成 unresolved。不要为此替换 AIAA Torch。BGE-M3
沿用现有 CPU adapter；Qwen3-Embedding-0.6B 使用固定的
`transformers 5.6.2 / huggingface-hub 1.12.0 / tokenizers 0.22.2` 轻量兼容层，继续复用 AIAA
`torch 2.12.1+rocm7.2`。本机已用哈希冻结的 0.6B 模型完成公开 Markdown 的 seekdb 导入和检索
烟测；这证明环境/协议可执行，不是检索质量 bake-off。

```bash
scripts/aiaa-protbind.sh -m protbind_agent knowledge inspect paper.pdf \
  --pdf-backend auto --ocr auto --confirm-data-access

scripts/aiaa-protbind.sh -m protbind_agent knowledge model-doctor \
  --embedding-model /reviewed/local/model
```

OpenFold3 启动入口为：

```bash
PROTBIND_OPENFOLD_GPU=0 scripts/aiaa-openfold3.sh validate-openfold3-rocm
```

该引导脚本不下载 checkpoint、CCD 或第二份 Torch。模型与 CCD 必须作为单独、显式、
带来源和 SHA-256 的预取步骤处理。

旧 ESMFold v1 也复用 AIAA 中的 `fair-esm 2.0.0` 和同一 ROCm Torch，但保持独立子进程、GPU
lease 与三权重 hash。它只在没有可复用受体时运行：

```bash
scripts/aiaa-esmfold-v1.sh python scripts/esmfold_v1_smoke.py \
  --model <local-esmfold-checkpoint> \
  --esm2-model <local-esm2-backbone> \
  --esm2-regression <local-contact-regression> \
  --environment-lock requirements/aiaa-esmfold-v1-overlay.lock.txt \
  --workspace <private-workspace> --receipt-output <private-receipt.json> \
  --sequence <protein-sequence> --device 0
```

三个权重缺一不可；该命令生成受体及 path-free receipt，不生成配体姿态。一次 24 aa
离线单卡实测成功，峰值分配显存为 `8,496,247,808` bytes（约 7.91 GiB），第二张卡未使用。

## 双 W7900 资源策略

- GPU 0：单个 OpenFold3 `p2-155k`、`low_mem`、ROCm Triton、一个 diffusion
  sample；启动前要求至少 28 GiB 空闲显存。
- GPU 1：HipFire 和交互界面；OpenFold3 运行时不在此卡启动第二个折叠作业。
- Vina/Meeko、PoseBusters、ProLIF、sPyRMSD 与 seekdb 使用 CPU。
- OpenMM 仅在当前构建真实暴露 HIP platform 时才走 HIP；本机 AIAA 当前只有
  `Reference` 和 `CPU`，因此当前不能产生 OpenMM HIP 证据。若后续重建出现 OpenCL，也不能把
  OpenCL 结果写成 HIP 证据。
- OpenMM 与折叠 worker 保持独立进程，避免平台初始化、显存和运行时状态互相污染；是否存在 HIP
  必须由该次 doctor/worker receipt 实测，不能由 ROCm 主机环境推断。
- 两张卡只分发独立查询，显存不能合并。结构缓存/RCSB 精确命中优先于折叠；schema-2 主状态为
  `... → SELECTED → DOCKED → VALIDATED ...`，只有通过 selection 的 top-8 才有资格进入可选
  OpenFold3 side task，避免每个候选都占用折叠资源。
- 不死磕单一共折叠器：OpenFold3 未通过真实 checkpoint 门禁时，旧 ESMFold v1 只负责
  生成受体；ESMFold2 只有通过 3 个公开复合物的离线 gfx1100 门禁后才能替代共折叠。
  两者都不可用时 schema-2 manifest 将可选 cofold 记为 `NOT_REQUESTED`、`UNAVAILABLE` 或
  `FAILED_RECOVERABLE`，继续 Vina→PoseBusters/ProLIF 主线，不伪造复合物预测，也不会仅因缺少
  cofold 而把主 run 标成 `DEGRADED`。

Vina worker 直接消费 `SELECTED` bundle。规范 docked pose 是 Meeko 恢复的 SDF，同时保留最佳/全部
PDBQT 和全部 modes SDF；pose-extraction 与 receptor-preparation receipt 精确绑定化学身份和受体
准备。`VALIDATED` 前由 host 从 `DOCKED` bundle 自动构建 validation batch 和本地 toolchain
attestation。独立 reference pose 只能在 `DOCKED` 后附加并标为 `VALIDATION_ONLY`。

## 依赖边界

覆盖层的直接版本（包括固定的 Dimorphite-DL 和 Qwen embedding 兼容层）在
`requirements/aiaa-protbind-overlay.txt`，实装传递依赖在
`requirements/aiaa-protbind-overlay.lock.txt`。官方 Vina 二进制的来源和 SHA-256
记录在 `tools/README.md`；二进制本身不进入版本库。

两个 requirements 都刻意不列 `torch`、`triton` 或 ROCm wheel。升级 Transformers 不是 Torch
不兼容的证据，也不授权 pip 替换共享 AIAA 的 GPU 栈。

OpenFold3 的小覆盖层分别由 `requirements/aiaa-openfold3-overlay.txt` 和
`requirements/aiaa-openfold3-overlay.lock.txt` 冻结。它刻意不列出 Torch、Triton、
ROCm、NumPy、RDKit 或 Gemmi；这些都必须来自 AIAA。

ESMFold v1 的小覆盖层由 `requirements/aiaa-esmfold-v1-overlay.txt` 和对应 lock
冻结，只补 `dm-tree`、`ml-collections`、`ihm`、`modelcif`。legacy OpenFold 从官方
源码固定到 commit `e938c184a291bf053af3b14c1e3e8bb29aee57e2` 并以 `--no-deps`
构建，避免 pip 解析或覆盖 AIAA Torch。

PowerMem 1.1.7 不进入核心覆盖层：其当前依赖会把 AIAA 的 NumPy 2.2.6 降到 1.x，
并引入多套云端 provider。P0 仍以 seekdb 为唯一精确状态源；PowerMem 后续放在独立、
无网络、只接收 seekdb artifact ID 的可选 worker 环境中验证。

## 可复现审计

以下命令生成不含绝对环境路径或密钥的 JSON：

```bash
scripts/aiaa-protbind.sh scripts/aiaa_environment_audit.py \
  --output experiment-results/aiaa-environment.json
scripts/aiaa-openfold3.sh python scripts/aiaa_openfold3_audit.py \
  --output experiment-results/aiaa-openfold3-environment.json
```

审计包括软件版本哈希、两张 Radeon、ROCm 版本、Vina SHA-256、OpenMM platform 和
资源策略。它不把任何探测值包装成科学性能结论。
