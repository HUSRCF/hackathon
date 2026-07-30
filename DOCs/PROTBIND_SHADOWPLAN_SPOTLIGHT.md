# ProtBind ShadowPlan Spotlight

状态：protocol revision 1，2026-07-30 冻结。

适用范围：内置 HipFire Agent、ProtBind MCP/OpenCode 适配器和本地 Web/CLI。
科学状态源：ProtBind manifest、stage-gate receipt 和 content-addressed artifact；ShadowPlan
从不成为科学状态源。

## 1. Spotlight

ProtBind ShadowPlan 是一个可撤销、可审计的推测式科研控制层：

> 当 Agent 等待用户批准或长时科学工具返回时，控制层只基于已经获准且不可变的状态，
> 预先编译条件式下一步、失败恢复路径和界面材料；它不会提前读取私有数据、联网、写入状态、
> 使用 continuation token 或臆造尚未产生的科学结果。

这不是“让模型在后台秘密继续执行”。它的目标是把不可避免的等待转化为有界、可取消且能出具
receipt 的准备工作，同时保持 ProtBind 的单阶段闭环：

```text
case_status
  -> human-visible action preview
  -> WAITING_APPROVAL + ShadowPlan
  -> permission reply
  -> snapshot/policy revalidation
  -> case_advance(fresh token, exactly one stage)
  -> ACCEPTED postflight
```

批准凭证与科学 continuation token 是两种独立凭证。ShadowPlan 不保存或预消费 continuation
token；批准后仍须重新验证 manifest 和 control-policy 绑定。

## 2. 当前性能基线

正式基线为：

- receipt：`experiment-results/protbind-agent-w7900-c58ca3c.json`
- ProtBind revision：`c58ca3cc224ad2ca0979ef148d2119e595da319f`
- HipFire revision：`92419d74e527caf1a283852ad5b059f70c0208f2`
- Qwen3.5 9B MQ4 weights SHA-256：
  `ba83acf5bfd5d4e334b0afc26d779734e31623bb7f74e807c3581dfecb3128ad`
- Radeon：W7900/gfx1100；HipFire speculation 关闭

三次 measured runs：

| 指标 | 结果 |
|---|---:|
| E2E wall p50 / p95 | 16.552 / 16.692 s |
| first-model TTFT p50 / p95 | 9.605 / 9.707 s |
| model time / wall | 84.0%–86.3% |
| tool time / wall | 13.7%–15.9% |
| tool sequence / success / artifact citation | 1.0 / 1.0 / 1.0 |
| selected-GPU peak used VRAM | 7,271,006,208 bytes |

第一轮请求报告 2,850 prompt tokens，第二轮报告 3,886 prompt tokens。内置 Agent 当前注册
18 个工具，完整 schema 的 canonical JSON 为 7,084 bytes，而正式 workload 只调用
`case_status`、`knowledge_search` 和 `memory_write`。因此 P0 优先减少前填充和返回上下文，
而不是先优化仅占约 14%–16% 的工具编排。

HipFire Qwen 路径采用上一轮对话的 LCP prompt cache。正式基线第二轮明显快于第一轮，但独立
ShadowPlan 大模型会话会形成分叉或更短 prompt，可能让单活跃前缀冷启动。protocol revision 1
禁止在同一 HipFire 模型实例中插入并发影子 LLM 会话。

## 3. 三条执行通道

```text
用户请求 / tool call
        |
        +-- Control lane
        |     permission、manifest/policy hash、fresh gate、postflight
        |
        +-- Science lane
        |     TriPharm、Vina、folding、PoseBusters、ProLIF、OpenMM
        |
        `-- Idle lane
              deterministic ShadowPlan、UI/report skeleton、approved cache warmup
```

Control lane 始终有最高优先级。Idle lane 必须可取消，且不能减慢 Science lane 超过验收门限。
在单卡平台上，Idle lane 默认只运行 CPU 确定性任务；双 gfx1100 平台仍不得让独立影子 LLM
破坏 HipFire prompt cache；四卡平台也只有在单独通过资源与科学一致性门禁后才能启用额外模型。

## 4. ShadowPlan 数据契约

每个计划至少包含：

```json
{
  "schema_version": "1.0",
  "kind": "protbind.shadow-plan",
  "status": "WAITING_APPROVAL",
  "plan_id": "sha256 without prefix",
  "tool": "case_advance",
  "arguments_sha256": "redacted canonical argument digest",
  "snapshot": {
    "manifest_sha256": null,
    "policy_sha256": null,
    "revalidation_required": true
  },
  "safe_idle_tasks": [
    "render-action-preview",
    "compile-conditional-branches"
  ],
  "forbidden_before_approval": [
    "private-data-read",
    "network-access",
    "scientific-state-write",
    "continuation-token-use",
    "memory-write"
  ],
  "branches": {
    "approved": "cancel idle work, revalidate bindings, execute the confirmed tool",
    "declined": "discard the ephemeral plan without state or memory writes",
    "stale": "discard the plan and request a fresh gate and confirmation",
    "tool_error": "record an explicit control failure and stop"
  }
}
```

`plan_id` 对不含生成时间的安全投影做 canonical SHA-256，因此同一 action preview 可重复验证。
原始参数、内部绝对路径、私有序列、API key 和 continuation token 不进入计划；只保存参数摘要。

## 5. 允许与禁止的 idle work

等待批准期间始终允许：

- 渲染 host 已提供的 action preview；
- 编译批准、拒绝、stale 和 tool error 的条件分支；
- 生成不含未完成结果的报告骨架；
- 罗列已经存在且已经获准的 artifact 引用；
- 计算本次计划、代码、配置和 prompt suite 的哈希；
- 更新本地、无私有内容的进度界面。

只有在同一授权范围已经明确成立时才允许：

- 复用已经打开的本地 seekdb/embedding 实例；
- 预热已经获准访问的索引；
- 为已经完成的 artifact 生成派生 UI 元数据。

等待批准期间禁止：

- 读取尚未获准的蛋白质、配体、论文或私有库；
- 发起 RCSB、UniProt、ColabFold 或任意其他网络访问；
- 写入 manifest、artifact、library、knowledge index 或长期经验；
- 调用 `case_advance`、附加 support 或复用旧 continuation token；
- 预判 docking/folding/validation 结果；
- 把计划、截图、检索或历史经验升级为科学证据。

用户拒绝后必须丢弃 ephemeral ShadowPlan。不得将拒绝本身或计划内容自动写入 PowerMem/experience。

## 6. P0 延迟优化

### 6.1 Deterministic Tool Pack

Host 根据显式工具名和高置信意图词选择最小工具包。模糊请求退回完整 allowlist，绝不使用另一次
LLM 调用做路由。模型只能执行当轮实际暴露的工具；即使 backend 返回一个已注册但未暴露的工具名，
host 也必须拒绝执行。

初始工具包：

- `case-control`
- `report`
- `knowledge-memory`
- `library`
- `public-fetch`
- `doctor`

路由只能减少 schema，不放宽 `SideEffect`、确认、文件、网络或科学状态权限。

### 6.2 Compact LLM Tool View

工具的完整 Python 返回值仍保留给 host；送入下一轮 LLM 的 projection 只包含：

- gate decision、state、next stage、failed checks 和 required actions；
- fresh continuation token，仅限正式 `case_status` 工具结果；
- acceptance/next-gate 摘要；
- 有界检索文本及页码/章节；
- artifact ID、warning 和科学语义。

projection 不得删除模型完成下一次合法调用所需的字段，也不得增加工具未测量的解释。

### 6.3 Persistent Knowledge Store

第一次明确数据访问确认之后，MCP service 可以按 workspace 和已哈希的 embedding manifest 复用
`SeekDBKnowledgeStore`。每次检索仍要求新的 `data_access_confirmed=true`，缓存不等于长期授权。
模型或索引身份变化时必须重建实例。

### 6.4 Prompt-cache discipline

- 每个活跃 case 保持稳定 system prompt 和工具包顺序；
- 禁止在同一 HipFire instance 上穿插独立 ShadowPlan 会话；
- 记录每轮 exposed tool count/schema bytes、prompt tokens 和 cache receipt；
- DFlash/MTP/ngram 只有在结构化工具调用、引用、状态门禁全部通过后才能晋级。

## 7. OpenCode 适配边界

OpenCode 是交互适配器，不是科学状态机。插件可订阅：

- `permission.asked` / `permission.replied`
- `tool.execute.before` / `tool.execute.after`
- `session.status` / `session.idle`

这些事件以 OpenCode 官方
[Plugins](https://opencode.ai/docs/plugins/) 文档为准；需要独立前端时，可使用官方
[Server SSE/OpenAPI](https://opencode.ai/docs/server/) 和
[SDK session/abort](https://opencode.ai/docs/sdk/) 接口。比赛冻结版本必须另记实际 OpenCode
版本和插件源码哈希，不能只引用滚动更新的在线文档。

插件职责：

- 展示 ShadowPlan、tool timeline 和批准范围；
- 在用户回复时取消 idle work；
- 将批准结果交回 ProtBind；
- 显示 adopted、declined 或 stale receipt。

插件不得直接读取私有目录、调用网络、执行 shell、签发 continuation token 或修改 manifest。
内置 Agent、CLI 和 Web UI 应消费同一 PlanAhead contract，避免把安全闭环绑定到 OpenCode。

## 8. 验收与证据

性能 benchmark 必须使用相同代码、模型、权重、输入、seed、prompt suite 和工具结果。至少进行
1 次 warmup 和 10 次 measured runs，分别报告：

- first verified UI card latency；
- first-model TTFT；
- approval-to-dispatch latency；
- complete E2E p50/p95；
- exposed schema bytes 和 prompt tokens；
- prompt-cache hit/cached tokens；
- tool、citation 和 stage-gate success rate；
- plan adopted/discarded/stale rate；
- idle work cancellation latency；
- GPU/CPU 占用和 peak VRAM。

硬门：

- tool sequence、tool success、artifact citation 必须保持 1.0；
- 未暴露工具执行次数必须为 0；
- 未批准私有读取、网络或状态写入必须为 0；
- stale continuation token 必须拒绝；
- ShadowPlan 不得改变 scientific output bytes；
- 单卡 Science lane slowdown 目标不超过 5%，否则关闭 idle work；
- 不宣称未经 A/B receipt 支持的加速百分比。

## 9. 分阶段交付

P0：

- 动态工具包和未暴露工具拒绝；
- compact LLM tool view；
- seekdb/embedding 实例复用；
- tool exposure/LLM latency receipt；
- 确定性 ShadowPlan contract 和 CLI action preview。

2026-07-30 protocol revision 1 的本地实现检查：

- 正式 workload 的静态 schema 从 18 tools / 7,084 bytes 路由为
  3 tools / 1,086 bytes，减少 84.7%；
- 路由关闭开关为 `--no-tool-routing`，供相同 workload 的 A/B 对照；
- host 保留完整 tool value，LLM 只接收 evidence-preserving compact view；
- seekdb/embedding store 在首次逐调用确认后惰性创建并在 MCP service 内复用；
- permissioned ToolSpec 会生成只含参数 SHA-256 的 ShadowPlan，并把 `plan_id` 写入 tool timeline；
- 374 项 pytest 全量回归和 `ruff check .` 通过。

上述 schema 数字是确定性输入规模检查，不是 TTFT 或 E2E 加速结论。新的 Radeon/HipFire A/B
receipt 生成前，正式性能基线仍是第 2 节记录的 16.552 s p50。

P1：

- 非阻塞 `WAITING_APPROVAL -> resume` runtime；
- snapshot/policy revalidation；
- OpenCode event plugin；
- Web/CLI plan timeline；
- controlled approval-delay benchmark。

P2：

- HipFire 多会话 KV slot；
- 真正逐 token SSE；
- 经 agentic conformance gate 的 speculative decoding；
- 额外 GPU 上的受资源约束小模型 ShadowPlan。

比赛 Spotlight 的证据口径固定为：

> ProtBind 在不越过隐私、权限和科学状态边界的前提下，把等待时间用于可撤销的确定性准备工作，
> 并用 hash-bound receipt 证明计划何时被采用、拒绝或因状态变化而丢弃。
