# ProtBind 运行 dossier 与姿态检查设计

## 结论

ProtBind 不把 PyMOL、截图或人工观感加入科学主链。执行状态由 manifest、stage record、artifact 和
stage-gate receipt 决定；姿态有效性由 PoseBusters、对称 RMSD、ProLIF/IFP 和受支持的物理检查决定。
本地 3D 视图只帮助研究者发现“应该回看哪一项工具证据”。

当前实现分成三层：

1. `case report`：完成 `REPORTED` 后的证据边界科学报告；
2. `case dossier`：任意 checkpoint 的执行/闭环/失败/产物账本；
3. `case pose-view` + loopback Web：Agent 可读无坐标摘要，人类可交互查看原始姿态。

## 为什么不把 PyMOL 编进 AIAA

对 AIAA 的 conda dry-run 没有修改环境，但解析结果显示 PyMOL 的 Qt/OpenGL/netCDF/HDF5 依赖链与
现有 AIAA pin 冲突。源码编译还会增加编译器、GUI、OpenGL 和平台插件维护面，却不产生新的
PoseBusters、IFP、RMSD 或物理证据。Bio.PDB 适合解析和结构分析，不是渲染器。因此当前组合为：

- Gemmi：PDB/mmCIF 原子与残基解析；
- RDKit：SDF/PDBQT 配体坐标和化学对象；
- 3Dmol.js：本地 WebGL 交互渲染与浏览器 PNG；
- FastAPI：仅回环地址提供选择后的 content-addressed receptor/pose。

曾审计 Proviewer 的公开仓库实现；它本质上也是 py3Dmol/Tkinter/Plotly 组合，并未调用 PyMOL。
该仓库未提供可识别的 LICENSE，因此 ProtBind 不复制其源码，只采用独立实现。

## Dossier 闭环语义

每个主阶段有以下互斥显示状态：

- `COMPLETED_ACCEPTED`：有 stage record，且存在绑定该阶段的 `ACCEPTED` postflight receipt；
- `COMPLETED_UNRECEIPTED`：计算记录存在，但闭环 acceptance 缺失；
- `NEXT` / `PENDING`：尚未运行；
- `BLOCKED_RETRYABLE` / `FAILED`：来自 manifest 的真实 failure record。

Dossier 输出 JSON、Markdown 和 HTML，记录：

- run/case/schema/state/next stage；
- 主阶段完成率与闭环 acceptance 率；
- 每阶段耗时、输入/config/cache SHA-256、输出、warning 和 acceptance receipt；
- optional cofold 状态；
- failures、required action 与 control receipts；
- 输入和命名 artifact inventory；
- pose validation 与无坐标几何 QA 摘要；
- 解释边界。

Continuation token、私有绝对路径、API key 和坐标字节不会进入 dossier。生成 dossier/scene 只会写入
内容寻址的派生 artifact，不修改 scientific manifest 或 stage state。

## 姿态查看数据边界

MCP/Agent 可见：

- candidate/molecule/engine、Vina 工具分数及其语义；
- receptor/pose ArtifactRef 和 docking box；
- PoseBusters、对称 RMSD、IFP、OpenMM 支持状态与 evidence artifact ID；
- Gemmi/RDKit 确定性计算的重原子数、最短跨分子距离、`<2 Å` pair 数、5 Å pocket residue 列表、
  ligand 是否位于声明 box；
- 本地 viewer 路径。

MCP/Agent 不接收 receptor/ligand 坐标正文。只有 `127.0.0.1`/`localhost`/`::1` Web UI 可按选中的
run/candidate 读取坐标，响应带 `private, no-store`。3Dmol.js 缺失、manifest 不匹配或 SHA-256
篡改时静态资源端点拒绝服务。

## 安装和使用

```bash
protbind assets install-3dmol \
  --approve-network cdn.jsdelivr.net \
  --workspace artifacts/protbind

protbind case dossier RUN_ID --format markdown
protbind case poses RUN_ID
protbind serve --workspace artifacts/protbind
```

在线安装只允许精确域名，固定 3Dmol.js `2.5.4`：

- JavaScript SHA-256:
  `1297081865a4d6c0b2ac22d3e909724da8c03ba0caf7bfc78c8a3d9d8b143f4e`
- LICENSE SHA-256:
  `4c6eaaed856f3f28a3b1a98e74f4a8a71618de7d51ea4155c29f6f793bcef861`

也可用 `--from-file` 和 `--license-file` 从已审核本地文件安装；仍执行相同哈希门。

## 后续值得加入的功能

P0：

- 用公开 canonical DOCKED + VALIDATED run 做 Chrome/Firefox 真实 WebGL 与 PNG 验收；
- 保存 viewer asset hash、camera、model selection 和 style 的截图 sidecar receipt；
- 在 pose 页逐项链接 PoseBusters/IFP/RMSD artifact，而不是只显示摘要；
- dossier 加机器可读的“下一项 required action”总表和 stage wall-time/worker-time 分离。

P1：

- 多候选叠合、reference pose overlay 和 symmetry mapping 显示；
- 可切换氢键/疏水/盐桥/π interaction overlay，数据必须来自 ProLIF artifact；
- receptor missing residue/altloc/repair provenance overlay；
- 将用户视觉批注保存为 `visual-review` artifact，但永远不改变 evidence grade。

P2：

- 无头浏览器生成可复现截图；只有在 browser/runtime/camera/style/hash 全冻结后才能用于审计附件；
- 对大型体系做服务端裁剪和渐进加载；
- 如果出现只有 PyMOL 才能满足的明确需求，再将其放入独立 worker/environment，而不是污染 AIAA
  Core。
