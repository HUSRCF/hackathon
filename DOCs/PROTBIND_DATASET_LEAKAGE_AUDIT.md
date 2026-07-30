# ProtBind 分子数据集泄漏审计

## 1. 目的

`protbind benchmark dataset-audit` 是所有虚拟筛选、化学空间泛化和 scaffold-novelty
主张之前的数据完整性门。它回答：

- 同一 split 内是否重复计数同一标准化 parent；
- 不同 split 是否共享标准化 parent、立体身份或重原子连接身份；
- 不同 split 是否共享 Bemis–Murcko scaffold；
- 不同 split 的 Morgan fingerprint 是否存在超过冻结阈值的高相似 analogue；
- 相似性检查是全量精确计算，还是因规模预算只能做确定性抽样。

它不回答：

- docking、ranking 或 screening 是否准确；
- 是否能富集真实 actives；
- ligand 是否实验结合；
- 蛋白序列、口袋、时间、assay 或标签是否泄漏。

即使所有检查均为 `PASS`，也只表示一组**数据完整性前置条件**通过。

## 2. CLI

```bash
protbind benchmark dataset-audit \
  --split train=data/train.smi \
  --split validation=data/validation.smi \
  --split test=data/test.smi \
  --dataset-name "example-screening-benchmark" \
  --dataset-version "2026-07-frozen" \
  --dataset-license "CC-BY-4.0" \
  --dataset-source "https://example.org/benchmark-v1.smi" \
  --similarity-threshold 0.8 \
  --max-similarity-comparisons 1000000 \
  --output artifacts/example-dataset-audit.json
```

输入支持：

- `.smi`、`.smiles`、`.txt`；
- `.csv`，列名为 `smiles` 或 `canonical_smiles`；
- `.sdf` / `.sd`；
- `.parquet` / `.pq`，需要 PyArrow。

至少需要两个命名 split，最多八个。所有计算离线完成，不发送分子或文件名到网络。
SMI、CSV、SDF 逐记录流式解析，Parquet 按 10,000 行 batch 解析；内存中只保留唯一
standardized parent 的紧凑身份表，不同时保留全部 RDKit Mol。

`dataset-source` 只保存公开 DOI/URL 或无路径 provenance label。绝对内部路径、含用户名/
密码的 URL、query token 和 fragment 会被拒绝。

## 3. 分子身份策略

每个可解析分子依次生成：

1. RDKit `Cleanup`；
2. `FragmentParent`；
3. stereo-aware canonical SMILES；
4. `ChargeParent`；
5. `TautomerParent`；
6. parent identity、non-isomeric connectivity identity；
7. Bemis–Murcko scaffold；无环分子使用确定性 element-graph key；
8. Morgan fingerprint，默认 radius 2、2048 bits、包含手性。

原始 SMILES 和 record ID 不写入 receipt。overlap example 只保存 canonical identity 或
scaffold 的 SHA-256 commitment。

该标准化策略有意偏保守：不同盐型、质子化或互变异构状态可能合并为同一 parent。它适合
检测潜在泄漏，但不声称与每个第三方数据集的官方 identity policy 完全相同。

## 4. 精确检查与规模预算

以下检查始终覆盖全部记录：

- parse/standardization failure；
- within-split duplicate parent；
- cross-split parent identity overlap；
- stereo identity overlap；
- connectivity identity overlap；
- scaffold overlap。

Morgan Tanimoto 需要计算 split 间笛卡尔积：

```text
comparisons = unique_parent_count(left) × unique_parent_count(right)
```

若不超过 `--max-similarity-comparisons`，状态为 `FULL`。若超过预算，按冻结 namespace
对 parent identity 做 SHA-256 排序，选取确定性子集，状态为
`PARTIAL_DETERMINISTIC_SAMPLE`。

部分审计遵循单向规则：

- 发现高相似 analogue：可以 `FAIL`；
- 未发现：只能 `INCOMPLETE`，不能 `PASS`。

因此在十万级或百万级集合上，如果要获得 analogue-novelty `PASS`，必须提高全量预算、
按 target/scaffold 合理分片后重新冻结协议，或后续接入具有精确召回保证的 fingerprint
索引后端。不能把抽样无命中解释成无泄漏。

## 5. Gate

receipt 输出五个门：

| 门 | PASS 条件 |
|---|---|
| `parsing_complete` | 每个 split 至少一个有效记录，且无 parse/standardization failure |
| `within_split_identity_uniqueness` | 每个 split 的标准化 parent 无重复记录 |
| `identity_novelty` | 任意 split pair 无 parent identity overlap |
| `analogue_novelty` | Morgan 检查为 FULL，且无非同一 parent 的高相似 pair |
| `scaffold_novelty` | 任意 split pair 无 scaffold overlap |

`broad_generalisation_precondition` 只有在上述五项全部 `PASS` 时才为 `PASS`：

- 任一确定失败优先产生 `FAIL`；
- 没有失败但存在部分审计时为 `INCOMPLETE`；
- 不会因为某些检查完成而掩盖另一个硬失败。

实际研究不一定要求 scaffold 完全不重叠。例如同 scaffold 的时间切分可回答特定问题；但此时
只能提出与该设计匹配的主张，不能声称 scaffold-new 泛化。

## 6. Positive control

仓库内置一个故意泄漏的 CC0 小型 positive control：

```text
benchmarks/fixtures/dataset-leakage-positive-control/
├── train.smi
├── test.smi
└── audit-receipt.json
```

其中：

- `CCO` 与 `OCC` 标准化为同一 parent；
- benzene 与 toluene 共享 Bemis–Murcko scaffold。

预期结果：

```text
parsing_complete                  PASS
within_split_identity_uniqueness PASS
identity_novelty                 FAIL
analogue_novelty                 PASS
scaffold_novelty                 FAIL
broad_generalisation_precondition FAIL
```

这个 fixture 证明审计器能检测已知泄漏，不是任何真实数据集的性能结果。

## 7. 与学术证据层的关系

建议 evidence ladder：

```text
dataset receipt
  ├─ FAIL       → 只允许数据修复、诊断或性能测试
  ├─ INCOMPLETE → 只允许风险提示，不允许 absence-of-leakage 主张
  └─ PASS       → 允许进入对应的后续实验，但仍不支持性能结论

frozen study protocol
  → model/workflow execution
  → statistical evidence packet
  → claim–evidence matrix
```

未来含 E4 external-generalisation 主张的 study protocol 应绑定：

- audit receipt SHA-256；
- split input SHA-256；
- threshold 与 fingerprint 配置；
- 对应 gate 名称；
- 蛋白 sequence cluster、时间和 assay leakage 的独立 receipt。

当前 fixed-ten known-site redocking pilot 明确将 generalisation 标为 `NOT_EVALUATED`，所以
不会用这个分子 split receipt 追溯性地晋升结论。

## 8. 下一步

protein sequence、pocket、PDB release-time 与 assay/label 已由
`benchmark research-leakage-audit` 形成四个独立 receipt，见
[`PROTBIND_CROSS_MODAL_LEAKAGE_AUDIT.md`](PROTBIND_CROSS_MODAL_LEAKAGE_AUDIT.md)。
仍需补齐可证明阈值召回的磁盘 fingerprint index、MMseqs2/PocketMatch 独立聚类 receipt、
RCSB metadata acquisition、真实集合许可复核，以及未来 E4 study protocol 的强制 support
binding。
