# ProtBind 私有数据仓库、P2Rank 与 DrutAI 接入

## 1. 两个独立库

蛋白质与小分子目录由操作者分别选择。ProtBind 不替用户猜保存位置，也不要求 root 权限：

```bash
protbind library init \
  --config .protbind/library.json \
  --protein-root /chosen/private/proteins \
  --ligand-root /chosen/private/ligands \
  --confirm-data-access
```

配置文件含本机绝对路径，权限设为 `0600`，且 `.protbind/` 已被 gitignore。不要把它放进报告或公开
提交。每个根目录包含：

```text
catalog.sqlite  objects/  incoming/  quarantine/  derived/  receipts/
```

原始文件按 SHA-256 不可变保存，RDKit 标准 parent 是单独 derived artifact，不覆盖原始分子。
导入成功的 `ACTIVE` 只表示本地解析/QC 成功；默认身份仍是 `UNVERIFIED`。

## 2. CLI 闭环

所有 library 子命令都要求 `--confirm-data-access`，便于脚本和 Agent 明确区分一次数据访问：

```bash
protbind library status --confirm-data-access

# 第一步仅扫描、哈希、冻结 plan，不导入
protbind library scan \
  --kind protein --source /existing/protein-batch \
  --recursive --confirm-data-access

# 审阅 plan ID 后复制；这是默认模式
protbind library import \
  --kind protein --plan-id <sha256> \
  --mode copy --confirm-data-access

# move 在 CAS 复制和 SHA-256 复核后才删源文件，并要求再次输入 plan ID
protbind library import \
  --kind protein --plan-id <sha256> \
  --mode move --confirm-move <same-sha256> \
  --confirm-data-access

protbind library list --kind protein --confirm-data-access
protbind library show --kind protein <entry-id> --confirm-data-access
```

扫描拒绝 symlink，限制文件数和单文件大小，只接受 PDB/mmCIF/FASTA 或
SDF/MOL/SMI/SMILES。apply 重新检查大小和哈希；重放同一 plan 返回原收据，不重复导入。解析失败
进入 `QUARANTINED`，超出 v1 范围则给出 `workflow_v1_compatible=false` 和 blocker。

UniProt 验证只发送用户给定 accession，不上传本地序列：

```bash
protbind library verify-uniprot <protein-entry-id> \
  --accession P12345 \
  --approve-network rest.uniprot.org \
  --confirm-data-access
```

输出区分 `EXACT_SEQUENCE`、`CONSISTENT_VARIANT`、`PARTIAL_COORDINATE_MATCH` 和 `CONFLICT`。
这些只描述观察序列与 accession FASTA 的一致性，不证明坐标、assembly、配体或结合真实性。

## 3. 蛋白库混合 RAG

库的 `catalog.sqlite` 仍是精确状态源；RAG 是可丢弃、可重建的 seekdb 投影：

```bash
protbind library rag-sync \
  --kind protein \
  --embedding-model /reviewed/local/bge-m3 \
  --confirm-data-access

protbind library rag-search "已做 UniProt 精确验证、少于 500 aa、无金属的条目" \
  --kind protein \
  --embedding-model /reviewed/local/bge-m3 \
  --confirm-data-access
```

投影只含 entry ID、状态、已核验 accession、格式、链/残基/缺失项计数和 workflow blocker；不含
文件名、绝对路径、序列、SMILES、分子字节或坐标。返回结果必须引用 snapshot artifact 与 entry
ID，再重新读取 catalog 条目并进入常规 QC/case gate；向量相似度不能证明身份、结构质量、结合或
活性。全文与向量两支都在 seekdb 查询内应用 `protein-library` scope，不做检索后的客户端过滤。

生产默认仍是 BGE-M3（1024 维、多语言且现有 AIAA/FlagEmbedding adapter 已接）。可选
`Qwen/Qwen3-Embedding-0.6B`，不是 0.8B；它必须有本地文件逐项 SHA-256 manifest，默认 CPU，
当前 AIAA 的 Transformers 4.48.1 低于本项目门禁 4.51.0，因此 `knowledge model-doctor` 会报告
`BLOCKED_RUNTIME_COMPATIBILITY`，不会自动升级 Torch/Transformers 或下载权重。

## 4. OpenCode/Codex Agent

默认 `opencode.json` 将 `.protbind/library.json` 交给本地 stdio MCP。配置不存在时 MCP 仍可启动，
但 library status 会要求操作者先运行 init。Agent 不接受任意路径，只能访问预配置的两个库及各自
`incoming/`。

八个 library MCP 工具（含 RAG sync/search）在 OpenCode 权限层全部是 `ask`。此外每次调用都要求
`data_access_confirmed=true`；`$protbind-library` skill 要求 Agent 先说明本次将读取、哈希、复制、
删除或联网的内容，再取得一次新的用户确认。一次确认不得复用到下一次调用。

这不是操作系统提权接口。ProtBind 不请求宽泛 sudo、setuid Python、`chmod 777` 或
NOPASSWD。若选定目录由管理员控制，操作者应在 Agent 外用最小范围的管理员动作创建目录并授予当前
用户；未来若提供 polkit helper，也只能白名单创建库根、探测文件系统与移动已验证导入等固定动作。

## 5. P2Rank

`protbind doctor` 检测本地 `prank`。CLI 可运行 P2Rank 或解析已有官方 predictions CSV：

```bash
protbind site p2rank-run \
  --receptor receptor.cif \
  --output-dir artifacts/p2rank \
  --bundle artifacts/p2rank-sites.json \
  --profile alphafold

protbind site p2rank-parse \
  --receptor receptor.cif \
  --predictions target_predictions.csv \
  --p2rank-version "P2Rank 2.5" \
  --bundle artifacts/p2rank-sites.json \
  --top-k 3
```

adapter 不经 shell 插值，限制 profile、timeout、输入格式及数值范围，并绑定 receptor 与
predictions SHA-256；直接运行要求稳定版 2.5，解析外部 CSV 时必须显式记录版本。输出是
`p2rank-site-hypotheses`，明确设置
`biological_site_validity_inferred=false`、`docking_box_validated=false`。它还不是
fpocket/P2Rank 共识自动调度；进入正式 ligand-only 流程前仍须实现共识、box 构造、受体坐标框架
检查和正常 docking 门禁。

## 6. DrutAI

DrutAI 已作为默认关闭的外部序列—SMILES一致性注释器接入，不进入候选淘汰或实验科学证据。
ProtBind 不复制上游实现，也不随源码、wheel、Docker 或提交材料分发 ONNX 权重。七个模型固定到
`HUSRCF/drutai_snap@5ee6ba7037466609edc06329782dee9298f20f2b` 的文件名、大小和 Git blob
commitment；首次获取要求批准 `raw.githubusercontent.com` 并精确确认 `GPL-3.0-only`，随后记录
观测 SHA-256、来源、curl 回执和私有缓存 artifact。

执行通过独立 `drutai.predict` 子进程和普通 TSV 通信，强制 `--no-cache`，私有序列/SMILES 在运行前
由 ProtBind fail closed 校验。普通二进制使用 bubblewrap `--unshare-net`；`/snap/bin` worker 则在
每次调用前核验 Snap 为 strict confinement、非 devmode/trymode 且没有已连接的 `network*` interface，
并把两份 Snap 元数据的 SHA-256 写入 bundle。任何检查失败都拒绝启动，避免 nested bubblewrap 剥夺
`snap-confine` capability。输出身份、顺序、概率范围和 0.5 分类规则必须完全匹配，否则整次调用失败。
结果只有 `SUPPORTIVE`、`DISCORDANT`、`ABSTAIN`；均为
`decision_eligible=false` 和 `hard_filter_allowed=false`。

已按项目维护者确认记录 ONNX 与 publication/original 输出一致，不重复将其列为 blocker。仍待完成的
科学门包括训练域与拆分泄漏记录、适用域/校准，以及公开 positive/inactive bake-off。因此 DrutAI
不能被称为结合 verifier，也不能升级结合、亲和力、活性、姿态或实验验证证据。
