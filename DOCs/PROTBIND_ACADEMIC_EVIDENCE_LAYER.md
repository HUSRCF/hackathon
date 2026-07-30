# ProtBind 学术证据层：从“流程可运行”到“主张可证伪”

## 1. 结论

ProtBind 已经具备较强的运行可追溯性：输入、配置、工具、输出、失败和阶段门禁均可由
hash-bound artifact 与 receipt 回查。当前更大的学术差距不是再增加一个预测器，而是把
**研究主张、实验设计、统计估计、负对照、消融和结论边界**绑定成一个可执行闭环。

本轮新增的 study evidence layer 将两类报告严格分开：

- case report / dossier：回答“一次运行做了什么、生成了什么、为何通过或失败”；
- academic evidence packet：回答“一组实验能够支持或反驳哪些预先声明的主张”。

这不是形式上的区别。单例对接成功不能推出方法有效，十例上的 90% 点估计也不能自动
推出广泛泛化；证据包必须显示分母、置信区间、配对检验、未运行对照和未评估主张。

## 2. 学术依据与 ProtBind 的口径

| 一手工作 | 对 ProtBind 的约束 |
|---|---|
| [PoseBusters, Chemical Science 2024](https://doi.org/10.1039/D3SC04185A) | RMSD 不能单独代表有效姿态；主要终点采用 `PB-valid AND symmetry-aware RMSD ≤ 2 Å`。 |
| [PoseBench, Nature Machine Intelligence 2025](https://doi.org/10.1038/s42256-025-01160-1) | 同时报告结构准确性、化学有效性和相互作用恢复；区分已知位点 redocking、blind docking、apo-to-holo 和多配体任务。 |
| [CASF-2016](https://doi.org/10.1021/acs.jcim.8b00545) | scoring、ranking、docking 与 screening power 是不同问题；ProtBind 不用 redocking 结果声称筛选命中率或结合亲和力。 |
| [LIT-PCBA](https://doi.org/10.1021/acs.jcim.0c00155) | 更接近真实类别不平衡的筛选评测需要实验活性/非活性集合，不能把 100k 性能库当成命中率实验。 |
| [2025 LIT-PCBA 数据审计预印本](https://arxiv.org/abs/2507.21404) | 该审计报告了重复、identity leakage 和 analog leakage。其结论仍应按预印本对待，但足以要求 ProtBind 在任何泛化实验前先生成去重与 split-similarity receipt。 |

因此，项目中的证据术语固定为：

- `SUPPORTED`：绑定 artifact 满足冻结规则；
- `CONTRADICTED`：绑定 artifact 明确不满足冻结规则；
- `INCONCLUSIVE`：方向或点估计可能有利，但统计或样本证据不足；
- `NOT_EVALUATED`：该主张没有被当前设计检验。

`SUPPORTED` 不是“生物学事实已证明”，也不是“实验结合已证明”。

## 3. 新的可执行闭环

新增实现：

```text
src/protbind_agent/study_evidence.py
benchmarks/studies/posebusters_fixed10_repair_v2_pilot.draft.json
benchmarks/studies/posebusters_fixed10_repair_v2_pilot.json
```

### 3.1 冻结协议

```bash
protbind benchmark study-freeze \
  --protocol benchmarks/studies/posebusters_fixed10_repair_v2_pilot.draft.json \
  --output benchmarks/studies/posebusters_fixed10_repair_v2_pilot.json
```

冻结动作会：

1. 校验唯一 primary claim；
2. 绑定 holdout SHA、selection hash、病例数以及 candidate/baseline regression manifest；
3. 固定 primary endpoint 为
   `PoseBusters-valid AND symmetry-aware heavy-atom RMSD <= 2.0 angstrom`；
4. 固定所有 frozen cases 进入分母，失败不得事后排除；
5. 固定 95% Wilson interval、双侧 exact McNemar 和 `alpha=0.05`；
6. 固定多重比较边界：一个主终点，其他只作次要或描述性结果；
7. 生成 `protocol_sha256`。

协议区分：

- `PROSPECTIVE_BEFORE_OUTCOME`：结果产生前冻结；
- `RETROSPECTIVE_AFTER_OUTCOME`：结果后补做的可复现分析。

后者不得使用 `CONFIRMATORY_BENCHMARK` scope，也不能被描述成追溯性“预注册”。单纯
的 Git commit 时间不能证明结果尚未被查看；正式前瞻研究还需要可信时间戳、签名或公开
登记。冻结协议与证据输出默认不可覆盖；只有明确指定 `--force` 才能重生派生文件，并且
必须保留旧 hash，不能把覆盖动作描述成同一 revision。

### 3.2 生成证据包

```bash
protbind benchmark study-evidence \
  --protocol benchmarks/studies/posebusters_fixed10_repair_v2_pilot.json \
  --candidate-results \
    experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/independent-regression.json \
  --baseline-results \
    experiment-results/posebusters-redock-fixed10-repair-protocol-v1-20260726/independent-regression.json \
  --output \
    experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/academic-evidence.json \
  --markdown \
    experiment-results/posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/academic-evidence.md
```

命令会 fail closed：

- protocol 自身 hash 不匹配则拒绝；
- regression 内部 `regression_sha256` 不匹配则拒绝；
- baseline/candidate 的 holdout hash 或 case identity 不一致则拒绝配对；
- 非 frozen-holdout 结果不得进入该比较；
- 缺失或失败病例按失败计数，不变更分母；
- 未提供 baseline 时，paired superiority 自动为 `NOT_EVALUATED`。

## 4. 当前十例 pilot 的真实结论

证据包：

```text
experiment-results/
  posebusters-redock-fixed10-repair-protocol-v2-restrained-sidechain-20260727/
    academic-evidence.json
    academic-evidence.md
```

| 项目 | protocol v1 | protocol v2 |
|---|---:|---:|
| 完成病例 | 9/10 | 10/10 |
| top-1 PB-valid + RMSD ≤ 2 Å | 8/10 | 9/10 |
| top-5 oracle | 8/10 | 9/10 |
| v2 的 top-1 95% Wilson interval | — | 0.596–0.982 |

配对病例中：

- candidate-only success：1（`7YZU_DO7`）；
- baseline-only success：0；
- top-1 绝对差：+0.10；
- exact two-sided McNemar `p=1.0`。

所以允许的表述是：

> 在固定十例 retrospective pilot 中，受约束的新原子侧链优化使一个原本无法完成的受体
> 进入可评估状态；v2 的完成率为 10/10，top-1 复现为 9/10。

不允许的表述是：

> v2 已被证明显著优于 v1，或 ProtBind 对一般蛋白—配体体系具有 90% 准确率。

当前 claim matrix 的结论：

- 十例上达到事后冻结的 80% 描述性阈值：`SUPPORTED`；
- 十例均完成：`SUPPORTED`；
- v2 配对统计优于 v1：`INCONCLUSIVE`；
- 泛化、亲和力和 screening hit rate：`NOT_EVALUATED`。

## 5. 证据阶梯

ProtBind 后续不应以“功能数量”衡量科研成熟度，而应沿下列阶梯升级：

| 等级 | 能回答的问题 | 当前状态 |
|---|---|---|
| E0 Artifact integrity | 输入、配置、代码、模型和输出能否被唯一回查？ | 已具备 |
| E1 Endpoint validity | 化学有效性、RMSD、IFP 是否按固定定义重算？ | 已具备 |
| E2 Internal reproducibility | 中断恢复、重复 seed、CPU/HIP parity 是否保持结论？ | 部分具备 |
| E3 Comparative evidence | 相同病例上的修订是否显著优于 baseline？ | 已有框架；当前十例不充分 |
| E4 External generalisation | 新序列、新 chemotype、apo receptor、预测位点是否仍有效？ | 未验证 |
| E5 Prospective utility | 能否富集真实 actives，或经湿实验确认？ | 未验证 |

比赛 Demo 可以以 E0–E2 为强项并展示 E3 的诚实结论；不得跳过 E3/E4 直接声称 E5。

## 6. 下一轮值得做的实验

### P0：冻结一份真正前瞻的回归协议

在查看新结果前：

1. 从 PoseBench 1.1+ 或其他许可明确的公开集合选择更大样本；
2. 冻结病例、工具版本、成功终点、排除条件和失败计数策略；
3. 使用外部可信时间戳或公开 release 证明冻结时间；
4. 把已知位点 redocking 与 blind/apo-to-holo 任务分开报告；
5. 先做样本量或最小可检测差异规划，再运行 candidate/baseline。

十例适合捕获工程阻断，不适合检验 5–10 个百分点的优越性。若大多数 paired cases
一致，McNemar 的有效信息只来自 discordant pairs，所需总病例数通常会进一步增加。

### P0：数据泄漏与数据集身份 receipt

分子 split 审计现已由 `protbind benchmark dataset-audit` 实现，输出：

- 原始记录数、canonical parent 数和 exact duplicate 数；
- train/validation/test 间 canonical identity overlap；
- Morgan fingerprint 最大跨 split Tanimoto 分布；
- target sequence identity / cluster；
- scaffold overlap；
- reference ligand 是否出现在候选库或训练集合；
- dataset version、license、source 与每个 split 的 SHA-256；
- 审计失败时禁止产生 generalisation claim。

当前实现对 identity、within-split duplicate 和 scaffold 做全量检查；Morgan 相似性超过
比较预算时只做确定性抽样，并把 absence-of-leakage 结论标记为 `INCOMPLETE`。完整命令、
规模语义与 positive control 见
[`PROTBIND_DATASET_LEAKAGE_AUDIT.md`](PROTBIND_DATASET_LEAKAGE_AUDIT.md)。
蛋白 sequence/pocket、PDB 时间、assay 与 label leakage 现已由
`benchmark research-leakage-audit` 输出四个独立 receipt；其 global-edit sequence metric、
外部声明 pocket cluster 和未联网复核的 PDB 日期均保留明确限制。详见
[`PROTBIND_CROSS_MODAL_LEAKAGE_AUDIT.md`](PROTBIND_CROSS_MODAL_LEAKAGE_AUDIT.md)。
下载时间和最终 URL 应来自另一个经过授权的 acquisition receipt；本地 leakage audit 不伪造
它未观察到的网络时间。

LIT-PCBA 可以作为规模或筛选方法研究来源之一，但在复核 2025 审计问题前，不应把它
单独作为“无偏泛化”证据。

### P0：最小负对照

当前协议已登记但尚未运行：

1. pocket box 平移：验证已知位点信息是否是成功的主要来源；
2. Vina pose ranking 冻结 seed 后打乱：分离 sampling 与 ranking；
3. ligand identity / bond order / stereochemistry tamper：确认错误化学不会通过 PB-valid；
4. no-repair baseline：分离“能进入 Meeko”与“姿态排名提高”。

负对照必须和主结果一起报告；不能只在主结果失败时补跑。

### P1：分层泛化矩阵

至少按以下维度报告，不训练一个不透明总分：

- receptor：holo / apo / predicted；
- site：known / P2Rank-fpocket consensus / blind；
- target novelty：sequence cluster 或 target family；
- ligand novelty：exact / scaffold-seen / scaffold-new；
- receptor chemistry：普通体系 / 金属 / 共价 / unsupported；
- task：redocking / cross-docking / virtual screening。

每个 cell 都保留样本数、成功数、区间和失败类型。样本不足的 cell 标记
`NOT_EVALUATED`，不能与其他 cell 池化后掩盖。

### P1：TriPharm 的科学消融

GPU parity 只能证明 HIP 与 CPU 实现一致，不能证明筛选有效。需要另行比较：

- TriPharm vs Morgan similarity；
- TriPharm vs shuffled ranking；
- ligand branch / pocket branch / equal-weight RRF；
- scaffold diversity on/off；
- known-site / predicted-site；
- top-512、top-128、top-16 各阶段 active recall 与 scaffold recall。

若使用真实 active/inactive 集合，报告 PR-AUC、ROC-AUC、EF1%、BEDROC 及置信区间；
类别高度不平衡时，不能只报 ROC-AUC。性能用的 100k 固定 ChEMBL 子集仍只回答吞吐，
不回答命中率。

### P2：与实验形成闭环

最强证据不是再加一个模型，而是：

1. 在算法冻结后选择一批未用于开发的可购化合物；
2. 记录候选选择、未选择候选和失败候选；
3. 盲态提交实验；
4. 回收 hit rate、剂量响应和 assay QC；
5. 把阴性结果同样写入 seekdb experience record；
6. 更新下一轮协议，但不得覆盖上一轮证据。

在此之前，报告应使用“computationally prioritised candidate”或“pose hypothesis”，
而不是“confirmed binder”。

## 7. Agent 如何使用证据层

Agent 可以：

- 解释 protocol、claim matrix、区间和失败原因；
- 建议下一组最有信息量的负对照或消融；
- 在等待批准时准备 study draft、样本量假设和数据审计计划；
- 检索历史相似失败并提示，但不自动改变冻结阈值；
- 生成带 artifact ID 的 reviewer-facing summary。

Agent 不可以：

- 在看过结果后把 retrospective draft 改名为 prospective；
- 因某病例失败而移出 frozen denominator；
- 自动选择更有利的 RMSD cutoff、top-k 或统计检验；
- 把 `SUPPORTED` 改写成“证明结合”；
- 用 viewer 截图替代 PoseBusters、RMSD 或 IFP；
- 用 Vina score 推断实验自由能。

这使 Agent 的“思考”成为研究设计辅助，而不是修改实验规则以迎合结果。
