# ProtBind 跨模态科研数据泄漏审计

## 1. 目标

`protbind benchmark research-leakage-audit` 为一个冻结 benchmark manifest 同时生成四个
独立、可验证的 receipt：

1. protein sequence-cluster leakage；
2. pocket artifact / declared pocket-cluster leakage；
3. PDB release-time leakage；
4. assay / replicate / target-compound label leakage。

四类结果不会平均成一个分数。任一已确定失败使
`broad_cross_modal_novelty_precondition=FAIL`；没有确定失败但序列比较不完整时为
`INCOMPLETE`；只有四项都通过时才为 `PASS`。

`PASS` 仍只代表严格的数据完整性前置条件，不代表模型有效、姿态正确、活性富集或生物学
结合。

## 2. Manifest

manifest 是私有本地 JSON，包含：

```json
{
  "schema_version": "1.0",
  "dataset": {
    "name": "...",
    "version": "...",
    "license": "...",
    "source": "https://..."
  },
  "split_roles": {
    "train": "TRAIN",
    "test": "EVALUATION"
  },
  "pdb_training_cutoff_date": "2022-12-31",
  "sequence_cluster_protocol": {
    "method": "...",
    "version": "...",
    "threshold_semantics": "...",
    "assignment_artifact_sha256": "..."
  },
  "pocket_cluster_protocol": {
    "method": "...",
    "version": "...",
    "threshold_semantics": "...",
    "assignment_artifact_sha256": "..."
  },
  "provenance": {
    "sequence_source_artifact_sha256": "...",
    "pocket_source_artifact_sha256": "...",
    "pdb_metadata_artifact_sha256": "...",
    "assay_metadata_artifact_sha256": "..."
  },
  "records": []
}
```

每条 record 必须具有：

```text
record_id
split
protein_sequence
sequence_cluster_id
pocket_artifact_sha256
pocket_cluster_id
pdb_id
pdb_release_date
assay_id
replicate_group_id
target_identity
compound_parent_identity
label
```

至少存在一个 `TRAIN` 和一个 `EVALUATION` split。缺失字段、非法蛋白字母、非 canonical
日期、重复 record ID、非法 SHA 或未声明 split 都会 fail closed。

manifest 可以包含私有序列和标签，但输出 receipt 不包含：

- 原始序列；
- 原始标签；
- record ID；
- assay、replicate、target 或 compound ID；
- 绝对内部路径。

示例只输出 SHA-256 commitment。

## 3. CLI

```bash
protbind benchmark research-leakage-audit \
  --leakage-manifest private/benchmark-manifest.json \
  --sequence-identity-threshold 0.30 \
  --max-sequence-comparisons 10000 \
  --output artifacts/research-leakage-receipt.json
```

默认不覆盖已有 receipt。只有明确指定 `--force` 才能重生派生结果，并应保留旧 hash。

## 4. Sequence-cluster receipt

当前内置 metric 明确定义为：

```text
global_edit_identity
= 1 - Levenshtein_edit_distance / max(sequence lengths)
```

这不是 MMseqs2 的 local sequence identity，也不能用 MMseqs2 的术语描述。
若环境存在 Biopython，使用其 C-backed `PairwiseAligner` 按 match=0、mismatch/gap=-1
计算整数 Levenshtein distance；否则使用等价的双行动态规划 fallback。backend 与版本写入
bundle implementation identity。

对每个 split pair：

- exact sequence SHA overlap 永远全量检查；
- hash-bound declared sequence cluster ID overlap 永远全量检查；
- 在预算内计算全部 unique-sequence pair；
- 超出预算时按 sequence SHA 做确定性抽样；
- 任一比较达到冻结 threshold 即 `FAIL`；
- 无命中但比较不完整为 `INCOMPLETE`；
- 只有 FULL 且无 threshold edge 时为 `PASS`。

manifest 同时绑定 sequence clustering method/version/threshold semantics/assignment SHA。
receipt 可以检查声明的 cluster ID 是否跨 split，但
`cluster_method_verification=NOT_EVALUATED`。如果研究主张要求标准的 30% local identity
cluster，应另行生成带 MMseqs2 binary/version/parameters/input/assignment hash 的 clustering
receipt，再将其绑定到 study protocol。当前 global-edit metric 和声明的 cluster ID 都不能
冒充已复核的 MMseqs2 结果。

## 5. Pocket receipt

口袋审计分两层：

### 5.1 Pocket artifact identity

`pocket_artifact_sha256` 相同表示两个 split 使用了完全相同的 pocket artifact。这是可直接
验证的字节身份，跨 split 重叠即失败。

### 5.2 Declared pocket cluster

`pocket_cluster_id` 来自 manifest 声明的外部聚类协议。receipt 绑定：

- method；
- version；
- threshold semantics；
- assignment artifact SHA-256。

它可以精确审计 cluster ID 是否跨 split，但不会重新计算 pocket embedding、PocketMatch、
结构比对或口袋几何。因此：

```text
cluster_method_verification = NOT_EVALUATED
```

真实泛化工作应另外保存聚类程序、模型、输入和 assignment receipt。仅在 manifest 中填入
不同 cluster ID 不能证明口袋真正不同。

## 6. PDB release-time receipt

冻结 `pdb_training_cutoff_date` 后：

- TRAIN：`pdb_release_date <= cutoff`；
- EVALUATION：`pdb_release_date > cutoff`；
- 任意 split pair 不得共享同一 PDB ID。

receipt 输出各 split 的日期范围、PDB overlap 和违规记录的哈希引用。

当前日期来自 hash-bound manifest/provenance artifact，没有实时访问 RCSB：

```text
rcsb_metadata_verification = NOT_EVALUATED
```

正式 temporal holdout 应先用授权的 RCSB acquisition receipt 固定 entry ID、初次发布日期、
obsolete/superseded 状态、最终 URL、获取时间和响应 SHA，再运行本审计。结构发布日期晚于
cutoff 也不自动排除相同复合物、序列或配体通过其他数据库进入训练集。

## 7. Assay / label receipt

严格 broad-novelty gate 检查：

- assay ID 跨 split 重叠；
- replicate group 跨 split 重叠；
- `(target_identity, compound_parent_identity)` 跨 split 重叠；
- 同一 entity pair 的冲突标签；
- 同一 split 内重复 entity-pair record。

所有原始 ID 和 label 均不写入 receipt，只输出数量和哈希示例。

这是一个严格 gate。对于 target-specific random split，同一个 assay 跨 train/test 可能是研究
设计的一部分；此时不能声称 assay-new generalisation，应冻结与该 estimand 匹配的较窄主张，
而不是为了让 gate 变绿而改写 ID。

该 receipt 不能发现：

- 不同 assay ID 实际来自同一实验板；
- 单位、阈值或曲线拟合不一致；
- PubChem/ChEMBL 间同一实验的别名；
- 文献重复、专利先验或 foundation-model 预训练泄漏。

这些需要上游 metadata reconciliation receipt。

## 8. Positive control

仓库提供一个故意同时触发四类泄漏的 CC0 fixture：

```text
benchmarks/fixtures/research-leakage-positive-control/
├── manifest.json
└── audit-receipt.json
```

它包含：

- train/test 完全相同的蛋白序列；
- 相同 pocket artifact 与 pocket cluster；
- 相同 PDB ID，且 evaluation release date 早于 cutoff；
- 相同 assay、replicate 和 target-compound pair；
- 同一 entity pair 的冲突 label。

预期结果：

```text
sequence_cluster FAIL
pocket_cluster   FAIL
pdb_temporal     FAIL
assay_label      FAIL
broad_cross_modal_novelty_precondition FAIL
```

该 fixture 是审计器 positive control，不是科学 benchmark 结果。

## 9. 与分子审计和 evidence packet 的关系

完整顺序：

```text
molecular split leakage receipt
  → sequence/pocket/time/assay leakage bundle
  → frozen study protocol
  → model/workflow execution
  → statistical evidence packet
  → claim–evidence matrix
```

两个 leakage receipt 互不替代：

- molecular receipt：标准化 parent、scaffold、Morgan analogue；
- research receipt：protein、pocket、PDB 时间和 assay/label。

只有与主张相关的所有 gate 通过，才允许开始相应的 external-generalisation 评测；之后仍需
报告样本数、失败分母、区间、负对照和统计比较。

当前 fixed-ten known-site redocking pilot 明确为 retrospective method verification，其
generalisation 为 `NOT_EVALUATED`。新 receipt 不会追溯性地把它晋升为外部泛化实验。

## 10. 后续增强

优先级：

1. MMseqs2 local-identity clustering worker 与独立 assignment receipt；
2. PocketMatch/结构比对或经验证 pocket embedding clustering receipt；
3. RCSB entry metadata acquisition 与 obsolete/superseded 追踪；
4. UniProt/PDB sequence mapping 和 chain-level release audit；
5. PubChem/ChEMBL assay alias、单位、阈值与 replicate reconciliation；
6. foundation-model pretraining corpus exposure 声明；
7. 将 receipt SHA 设为未来 E4 study protocol 的强制 support binding。
