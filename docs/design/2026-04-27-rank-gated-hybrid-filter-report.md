# 冻结语义基线下的 Rank-Gated Hybrid Filter 场景检索实验报告（草稿）

维护者：`zhuangbie@qq.com`  
日期：`2026-04-27`

## 摘要

本文面向道路场景图像检索任务，围绕“在不改动既有语义检索基线的前提下，能否通过词法过滤获得更少但更准确的结果”这一问题，设计并实现了一套 `Rank-Gated Hybrid Filter` 实验方案。该方案以既有 `Qwen3-VL-Embedding-2B + FAISS` 检索结果为冻结语义基线，以由既有 caption 文本确定性派生的 `BM25` 词法视图为辅助信号，通过 rank gate 交集规则对语义候选进行严格过滤。实验中，caption 结果文件、caption 提示词、embedding 模型及其检索配置均保持不变，仅在 hybrid 支路新增只读派生语料、过滤规则和对比输出目录。本文给出实验背景、问题定义、方法设计、实现约束、实验配置与评价协议，为后续人工评审和结果分析提供统一文档依据。

## 1. 引言

在当前项目的场景检索链路中，文本到图像检索的主方案已经收敛为基于 caption 文本的语义检索，即将查询文本与图片描述文本分别编码为向量，并通过相似度检索召回对应图片。该方案在已有实验中表现出较好的稳定性与可复现性，因此被视为当前阶段的正式语义基线。

然而，已有语义基线在若干 query 上仍会返回一定数量的边缘相关样本。对于面向人工审核的应用场景，仅依赖固定 `top_k` 的语义召回并不总能满足“结果更准”的需求。特别是在评审者更关注结果集合的语义纯度，而非召回数量时，允许输出数量下降、以换取更高精度，是一种合理且值得验证的实验方向。

基于上述背景，本文不再尝试通过重排扩大候选或改写原始语义排序，而是提出一种更保守的混合检索策略：在冻结 embedding 基线的条件下，将 BM25 仅作为硬过滤信号使用，从语义候选中筛除词法上不够贴题的结果。该策略的核心目标不是提高召回，而是在保持基线链路不变的前提下，验证 strict filtering 是否能带来更高的人工判读准确性。

## 2. 问题定义与实验目标

### 2.1 问题定义

给定一个查询文本 `q`，既有系统首先通过冻结的 embedding 检索链路，从 caption 语料中返回语义相似度最高的 `top-20` 图片候选。本文关注的问题是：在不修改该语义基线的模型、参数与结果文件的前提下，是否可以引入一条词法过滤支路，对这些候选进行二次筛选，使最终返回结果数少于 `top_k`，但整体更贴近 query 语义。

### 2.2 实验目标

本实验旨在回答以下研究问题：

1. 在冻结的语义检索基线之上，引入基于 BM25 的 rank-gated 过滤策略后，是否能够得到更小但更准确的结果集合。
2. 当系统允许输出数量低于固定 `top_k` 时，词法约束是否能有效削弱语义检索中的边缘相关样本。
3. 在缺乏人工逐图 GT 标注的条件下，如何以人工看片为主的方式，对过滤型 hybrid 检索进行可追溯评估。

### 2.3 非目标范围

为避免实验结论被其他变量污染，以下内容明确排除在本实验之外：

- 不修改 `caption` 结果 JSON；
- 不修改 `caption` prompt 与 prompt 配置；
- 不修改 embedding 模型、embedding 指令或 embedding 检索 CLI 默认配置；
- 不重跑 embedding 基线以获得新的语义排序；
- 不将 BM25 用作重排信号；
- 不将本实验与既有 hybrid 重排尝试混合分析。

## 3. 方法

### 3.1 总体框架

本实验采用“冻结语义基线 + 词法硬过滤”的混合检索结构。其数据流如下：

```text
冻结 caption JSON -> 确定性 BM25 lexical view -> BM25 排名
冻结 embedding 报告 -----------------------> embedding 排名
embedding rank gate + BM25 rank gate -----> 最终过滤结果
最终过滤结果 ------------------------------> 新报告与对比目录
```

与传统 hybrid rerank 不同，本文方案不对 surviving items 再做重排，BM25 仅用于判定“保留”或“剔除”。

### 3.2 冻结语义基线

语义基线采用既有 `qwen3vl_embedding_faiss` 后端，其核心过程为：读取 caption 结果文件中的描述文本，以 `Qwen3-VL-Embedding-2B` 将 query text 与 caption text 编码为向量，再通过 FAISS 进行近邻检索。需要强调的是，该链路在本项目当前实现中属于“文搜文”，即 query 检索的是 caption 文本，而非原始图片向量。

在本实验中，该语义链路的结果文件被视为不可变输入，hybrid 仅消费其产出，不对其重新建索引、改参数或覆盖结果。

### 3.3 BM25 词法视图

BM25 支路不直接使用原始 caption 长文本，而是从冻结 caption 语料中确定性构造一份只读 lexical view。该视图的构造遵循以下原则：

- 不引入新的模型推理；
- 不回写原始 caption 文件；
- 仅做确定性文本清洗与压缩；
- 尽量抑制时间戳、通道标记、标题式字段等检索噪声；
- 尽量保留与场景判别相关的短词或短语，例如低光、强光、前车、湿路、反光、刹车灯等。

词法视图以独立 JSON 文件保存，每条记录包含：

- `image_id`
- `image_path`
- `source_caption`
- `bm25_text`

该设计保证了 BM25 的输入可审计、可回溯，并与生产 caption 文件严格隔离。

### 3.4 Rank-Gated 过滤规则

设冻结语义基线对 query `q` 返回的候选集合为 `C_q`，其中：

- `C_q`：embedding `top-20`
- `S_q`：`C_q` 中 embedding rank 不高于 `10` 的子集
- `L_q`：BM25 rank 不高于 `8` 的集合

则最终过滤结果定义为：

```text
R_q = S_q ∩ L_q
```

并满足以下输出约束：

- 最终输出保持 embedding 原始顺序；
- 不对保留下来的样本再做重排；
- 最大输出数为 `10`；
- 最小输出数为 `0`；
- 若交集为空，则返回空结果；
- 不使用更弱候选补齐结果数。

该规则的核心思想是：语义检索负责给出可信候选头部，BM25 仅作为严格词法门控，以减少边缘相关结果。

### 3.5 设计动机

本文采用 rank gate 而非 score threshold，主要出于以下考虑：

1. embedding 分数在不同 query 间不一定具备良好可比性；
2. BM25 原始分数更容易受 query 长度与词频分布影响；
3. 基于 rank 的门槛更直观，便于人工审核与实验复现；
4. `top-10 ∩ top-8` 的交集规则足够保守，有助于保证 hybrid 不会无约束扩张候选集合。

## 4. 实验设置

### 4.1 数据与查询集

本实验使用当前 `issue_*` 场景数据集及其既有 caption 结果。语料统计如下：

- 图片总数：`106`
- caption 总数：`106`
- query 数量：`3`

查询文件为：

- `input/processed/scene_queries_specific_en.json`

其中文 query 用于展示与人工判读，英文 query 用于实际检索输入。三条查询对应的中文场景语义为：

- 低光照下，前车强逆光
- 对向来车
- 雨天，前车刹车灯亮起

### 4.2 冻结输入与实验约束

本实验明确冻结以下输入：

- 冻结语义基线报告：`output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json`
- 冻结 caption 语料：`output/reports/caption_issues_all_qwen35.json`
- embedding 模型：`Qwen3-VL-Embedding-2B`
- embedding 指令：`Represent the user's input.`
- caption prompt 及其配置

换言之，实验仅允许在 hybrid 支路新增文件和目录，不得改写 baseline 链路的任何产物。

### 4.3 检索后端与参数配置

本实验所涉及的核心配置如表 1 所示。

| 配置项 | 取值 |
|---|---|
| 语义后端 | `qwen3vl_embedding_faiss` |
| 词法后端 | `BM25` |
| 语义来源 | 冻结 embedding 报告 |
| 词法语料 | 冻结 caption 的确定性 lexical view |
| 语义候选上限 `candidate_top_k` | `20` |
| 语义门槛 `semantic_top_n` | `10` |
| 词法门槛 `lexical_top_n` | `8` |
| 最终输出上限 `output_max_k` | `10` |
| query 扩展 | 关闭 |
| 评价方式 | 人工看片为主，无 GT 自动指标 |

### 4.4 输出文件与目录结构

为保证实验可审计、可复查，本文将产物划分为报告文件、BM25 语料文件、最终结果目录和对比目录四类：

- 过滤报告 JSON：`output/reports/retrieval_issues_all_hybrid_filter_rank_gate.json`
- 过滤摘要 CSV：`output/reports/retrieval_issues_all_hybrid_filter_rank_gate_recall_summary.csv`
- BM25 lexical view：`output/artifacts/bm25_views/hybrid_filter_rank_gate.json`
- 最终图片目录：`output/retrieval_output_hybrid_filter_rank_gate/`
- 对比目录：`output/comparisons/hybrid_filter_rank_gate/`

其中，每个 query 的对比目录包含以下子目录：

- `embedding_top20/`
- `embedding_top10/`
- `bm25_top8_candidates/`
- `final_filtered/`
- `filtered_out_from_embedding_top10/`
- `manifest.json`

该目录结构服务于后续人工审核：评审者可以直接比较“语义头部结果”与“最终过滤结果”的差异，并检查哪些样本被剔除。

### 4.5 评价协议

由于当前实验未构建人工逐图 GT 标注，因此本研究不使用传统的 `Recall@K`、`Precision@K` 作为正式结论依据，而采用人工看片作为主评审方式。具体流程如下：

1. 以 `embedding_top10/` 作为冻结语义基线头部结果；
2. 以 `final_filtered/` 作为 hybrid 过滤后的最终结果；
3. 通过 `filtered_out_from_embedding_top10/` 观察被过滤掉的样本是否确属边缘相关或离题；
4. 仅将 `bm25_top8_candidates/` 视为辅助证据，而非最终判断标准。

在该协议下，实验关注点不再是“召回更多结果”，而是“在允许结果变少的情况下，是否显著提升结果集合的贴题程度”。

## 5. 实验结果

本节留空，待人工评审完成后补充。

## 6. 局限性与有效性威胁

尽管本实验通过冻结基线与隔离 hybrid 支路，尽可能减少了外部变量干扰，但其结论仍受到以下因素限制：

### 6.1 缺乏人工逐图 GT

当前评估以人工看片为主，尚未建立可量化、可复算的逐图真值标注。因此，本实验更适合作为精检索方向的可行性验证，而非终局统计结论。

### 6.2 BM25 仍可能保留场景噪声

尽管 lexical view 已做确定性清洗，BM25 仍可能对“夜间强光”“路面反射”等高频视觉线索赋予较高权重，从而保留并非真正目标场景的样本。

### 6.3 Rank Gate 可能过于保守

`embedding top-10 ∩ bm25 top-8` 的交集规则本质上偏向高精度、低召回。一旦 query 本身较模糊，或 BM25 对该 query 的区分能力不足，最终结果可能过少，甚至为空。

### 6.4 结果顺序未被重新优化

本文实验仅研究“保留谁、删除谁”，不研究 surviving items 之间是否还需进一步重排。因此，即使过滤结果整体更纯，集合内部排序质量仍然沿用冻结 embedding 基线。

## 7. 结论

本节留空，待实验结果与人工评审意见汇总后补充。
