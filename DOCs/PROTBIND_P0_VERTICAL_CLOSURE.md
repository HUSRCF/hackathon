# ProtBind P0 Agent/GPU/记忆垂直闭环

## 结论

P0 现在由四个相互绑定、但科学语义分离的组件组成：

```text
HipFire/Qwen 本地 Agent
→ 固定 ToolSpec + 逐调用确认
→ stage gate / scientific workflow / HIP screening
→ artifact-bounded report + deterministic experience
```

内置 Agent 不依赖 OpenCode，也不发现任意 MCP、shell、文件系统或网络工具。OpenCode 仍可作为
可选交互前端；两条路径复用 `ProtBindMCPService` 的同一门禁和 workflow。

## 1. 内置 Agent

入口：

```bash
scripts/aiaa-protbind.sh -m protbind_agent agent \
  --backend hipfire \
  --model qwen3.5:9b \
  --workspace artifacts/protbind \
  --project-root . \
  --knowledge-model .protbind/models/qwen3-embedding-0.6b \
  "请保持离线，检查案例状态并生成带 artifact 引用的摘要"
```

无需确认的工具只读取 doctor、case 状态/报告/dossier/pose QA、artifact metadata、embedding
状态或历史经验提示。public fetch、case create/advance/attach、library import、knowledge
import/search/RAG sync 和 memory write 每次调用都显示：

- 将读取和写入什么；
- 是否联网；
- 是否改变科学状态及预计下一阶段；
- 失败后的恢复语义。

确认由 host 注入，模型不能把 `data_access_confirmed=true` 写进参数绕过。每个
`case_advance` 仍须先 `case_status`，使用绑定最新 manifest 的新 token，只推进一个阶段，并得到
`ACCEPTED` 才可继续。

Agent 使用流式 completion，按 index 聚合碎片化 tool call；timeline 保存工具名、确认、耗时、
成功和错误类型，不保存私有参数或正文。达到最大步数时错误只返回工具名轨迹，不伪造最终答案。

## 2. Radeon Agent benchmark

benchmark 固定测试真实工具负载，而不是普通聊天：

```text
case_status
→ knowledge_search
→ memory_write
→ 带 sha256: artifact 引用的最终摘要
```

运行前必须准备一个深审计通过的公开 `REPORTED` run，并把公开 benchmark 文档导入同一 workspace。
正式命令至少重复三次、预热一次：

```bash
scripts/aiaa-protbind.sh -m protbind_agent agent-benchmark \
  --workspace artifacts/agent-benchmark \
  --run-id <reported-public-run> \
  --knowledge-query "ProtBind scientific boundary" \
  --preference "优先离线运行" \
  --knowledge-model .protbind/models/qwen3-embedding-0.6b \
  --model qwen3.5:9b \
  --model-revision <revision> \
  --model-weights .protbind/models/qwen3.5-9b.mq4 \
  --model-sha256 <sha256> \
  --quantization mq4 \
  --hipfire-source-root <clean-hipfire-checkout> \
  --hipfire-revision <git-revision> \
  --hipfire-daemon <exact-running-daemon> \
  --hipfire-visible-device 1 \
  --hipfire-speculation off \
  --hipfire-jinja-mode default-on \
  --code-revision <clean-protbind-revision> \
  --label w7900-gfx1100 \
  --output benchmark-results/protbind-agent-w7900.json \
  --confirm-benchmark-data
```

receipt 会重新测量模型权重、daemon 和已加载 `libamdhip64` 的 SHA-256，验证 health PID 的实际
进程树、GPU visibility、model、speculation/Jinja 配置、两个 Git revision/cleanliness、
gfx1100、HSA override、prompt suite、TTFT、吞吐、总时延、系统 Radeon 显存峰值、工具成功率、
调用顺序及 artifact citation。AIAA Torch/ROCm 版本与 HipFire 实际加载的 HIP runtime 分开报告；
不能把二者写成同一个版本。

只有所有 measured repetitions 的工具序列、工具成功和引用均通过，两个源码 checkout 干净，
进程/daemon/runtime 绑定可验证且有显存样本时，`evidence_eligible=true`。DeepSeek 不接受为该
benchmark backend。

2026-07-30 的正式 W7900/gfx1100 receipt 为
`experiment-results/protbind-agent-w7900-c58ca3c.json`，文件 SHA-256 为
`8b3f16fd63e54ac4d158d8945095ee9a2181a0e0ce43195f470e828a0ddd9ab6`。它绑定 ProtBind
`c58ca3cc224ad2ca0979ef148d2119e595da319f`、HipFire
`92419d74e527caf1a283852ad5b059f70c0208f2`、Qwen MQ4 权重
`ba83acf5bfd5d4e334b0afc26d779734e31623bb7f74e807c3581dfecb3128ad` 和实际加载的
HIP 7.14 runtime library。三次 measured run 的工具序列、工具成功和 artifact 引用通过率均为
1.0；总时延 p50/p95 为 16.552/16.692 s，首模型 TTFT 为 9.605/9.707 s，端到端模型吞吐为
33.139/37.138 tokens/s，所选 GPU 的峰值已用显存为 7,271,006,208 bytes。

HipFire 当前把每次响应近似作为单个流 span 交付，所以 receipt 将 post-first throughput 标为
不可测，而不是用接近零的 `total−TTFT` 生成虚高数字。该次正式运行关闭 speculation；此前 DFlash
开发烟测未能可靠完成结构化工具语法，不能混入正式结果。模型上游转换 revision 未随 MQ4 文件提供，
因此以完整文件 SHA-256 作为本地内容 revision；这足以复现本地字节，但不补写未知的上游 commit。

## 3. TriPharm HIP 生产接入

worker 配置：

```toml
[screening]
backend = "auto"
hip_executable = "build/tripharm_hip/tripharm_hip_query"
parity_top_k = 512
hip_timeout_seconds = 600
```

生产 adapter 从真实 persisted SQLite index 流式导出三角形和 query，HIP kernel 产生候选 mask，
CPU 对 HIP 候选执行同一 exact ranking；同时运行不受限 CPU reference。只有完整 top-k
`molecule_id` 顺序完全一致才提交 `committed_backend=hip`。`auto` 遇到不可用或 parity 失败会写
明确原因并提交 `cpu-reference`；`backend=hip` 则产生可恢复 stage failure。报告始终把分数称为
药效团几何匹配，不称结合分。

该 production 模式故意包含 CPU reference 作为正确性门，因此不能把全路径耗时包装成纯 kernel
加速。100k 的 kernel/transfer 和端到端 gate 成本必须分别报告。

真实 persisted-index 小型 smoke 保存在
`experiment-results/protbind-tripharm-hip-production-smoke-gfx1100.json`，文件 SHA-256 为
`8cce9523ac59c5ad71219eee536e9332d0bf99b5839708f1a194b33d3ab242ca`。它使用 3 个合成分子、
9 个 candidate triangles 和 4 个 query triangles，在第二张 W7900/gfx1100 上完成 1 次预热和
3 次测量；CPU/HIP 两个结果 ID 的完整排序哈希相同，三次均
`committed_backend=hip`。端到端 p50/p95 为 0.1376/0.1396 s，kernel p50 为 31.04 µs。
该夹具只证明编译产物、真实 index/query 协议和 parity gate 可执行，不是 100k 吞吐或 5×
加速证据。

## 4. 经验记忆

`memory_write` 只接受 `run_id` 和可选用户偏好。候选、evidence grade、失败码、工具版本和 artifact
IDs 均从深审计通过的 `REPORTED` manifest 确定性派生，写为内容寻址 experience artifact，再更新
SQLite 检索投影。seekdb 仍是文献和科学状态源；经验投影不能替代 manifest/artifact。

`memory_search` 可按相似受体标识、哈希化配体身份、失败原因、工具链和历史验证结果返回提示，但
返回值明确禁止自动复制 box、seed、阈值或科学结论。PowerMem 暂不进入 AIAA 核心环境，避免其依赖
替换 NumPy/GPU 栈。

## 5. 尚未由 P0 声称完成

- 100k 固定化学库上的正式 TriPharm 5× 性能门；
- Web 写操作面板、top-5 pose 比较和 fpocket/P2Rank 自动共识；
- OpenFold3 checkpoint、OpenMM HIP、ESMFold2 和多 GPU 调度。

这些不影响内置 Agent、确认门、HIP parity、确定性记忆和可追溯报告的代码闭环，但必须在演示或
评分材料中保持为未完成项。
