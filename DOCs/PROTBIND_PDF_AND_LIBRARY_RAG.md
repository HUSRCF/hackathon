# ProtBind PDF、OCR 与蛋白库 RAG

## 1. OpenCode 的边界

OpenCode 1.18.8 的内建工具表没有 PDF 文字提取或 OCR。它可以把图片交给支持视觉输入的模型，也能
加载 custom tool/MCP，但这不等于可审计的论文 PDF 解析。ProtBind 因此不向 Agent 开放 bash 或任意
文件读取，而是在本地 stdio MCP 中提供三个有界工具：

- `knowledge_document_inspect`：只返回提取能力、页数、扫描页和 warning，不返回正文；
- `knowledge_import`：导入项目根内的 PDF/Markdown，保存原文 artifact、逐页 chunk 和 extraction
  receipt；
- `knowledge_search`：按 `evidence|protein-library|ligand-library` scope 做全文+向量 RRF，只返回
  引用片段，不生成结论。

三个工具在 OpenCode 权限层均为 `ask`，每次还要求新的 `data_access_confirmed=true`。Agent 不能
传任意模型目录；操作者必须在 MCP 启动时配置 `--knowledge-model`。

## 2. PDF 提取门禁

```text
PDF 大小/页数上限
→ PyMuPDF 逐页原生文字
→ auto 时与 pdftotext -layout 逐页比较
→ 低于文字量阈值的页标记为 scan-like
→ OCR off / auto / required
→ 逐页 chunk + one-based page citation
→ 内容寻址 extraction receipt
→ seekdb import
```

`pdftotext` 适合含文字层的 PDF，不是 OCR。Tesseract 不直接读取 PDF；ProtBind 只在本机存在
Tesseract 时把受限页渲染成临时 PNG 后 OCR。`required` 遇到缺 Tesseract、超 OCR 页数或 OCR
失败会停止；`auto` 会把未解析页列入回执，不会静默把空页当成功。

```bash
protbind knowledge inspect paper.pdf \
  --pdf-backend auto --ocr auto \
  --confirm-data-access

protbind knowledge import paper.pdf \
  --embedding-model /reviewed/local/bge-m3 \
  --pdf-backend auto --ocr off \
  --license "reviewed local copy" \
  --confirm-data-access

protbind ask "这篇文献怎样定义 pose validity？" \
  --embedding-model /reviewed/local/bge-m3 \
  --scope evidence --top-k 5 \
  --confirm-data-access
```

回答必须引用 source artifact ID 与 PDF 页码（Markdown 使用 section）。扫描未解析页意味着检索
覆盖不完整；无命中不能解释为文献没有讨论。

## 3. 蛋白库 RAG

`catalog.sqlite` 是库状态源。每次 `rag-sync` 从 catalog 生成新的内容寻址 snapshot，先清除对应
seekdb scope 的旧投影，再写入当前记录，避免身份验证或 QC 更新后仍检索到陈旧文本。投影明确排除：

- 文件名、绝对路径和源目录；
- 蛋白序列和坐标；
- SMILES、分子字节和标准化分子值。

只保留 entry ID、状态、验证状态/已验证 accession、格式、链/残基/缺失项计数、普通体系兼容性与
blocker。检索结果用于找候选条目，不用于推断结合、活性、结构质量或真实身份。选中后必须重新
`library show` 并通过常规 case gate。

## 4. Embedding 决策

默认采用 `BAAI/bge-m3`：现有 adapter、1024 维、多语言、AIAA 中的 FlagEmbedding 均已具备。
`Qwen/Qwen3-Embedding-0.6B` 是可选模型；不存在官方“0.8B”这一档。两者都要求：

- 完全本地目录；
- `protbind-model-manifest.json` 固定 model name/revision；
- manifest 中每个文件逐项 SHA-256；
- `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`；
- 默认 CPU，避免占用折叠、TriPharm 或 OpenMM 的 gfx1100 资源。

当前 AIAA 的 Transformers 4.48.1 未通过 Qwen 的 4.51.0 runtime 门，状态为
`BLOCKED_RUNTIME_COMPATIBILITY`。这不是 Torch 缺失；ProtBind 不会自动安装 Torch、升级共享 AIAA
环境或下载模型。后续应在独立轻量 overlay 中做相同语料的 recall、中文/英文检索质量、延迟、RAM
和索引稳定性 bake-off，通过后才切换生产模型。

预取和审核权重后，由操作者冻结本地目录；该命令只哈希，不加载模型，也不联网：

```bash
protbind knowledge model-freeze \
  --embedding-model /reviewed/local/bge-m3 \
  --model-name BAAI/bge-m3 \
  --model-revision <exact-upstream-revision>

protbind knowledge model-doctor \
  --embedding-model /reviewed/local/bge-m3
```
