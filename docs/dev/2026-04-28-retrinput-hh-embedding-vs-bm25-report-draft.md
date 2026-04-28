# RetrInput_hh 场景检索实验报告

日期：2026-04-28

## 结论先行

对于“低光照强逆光”“对向来车”这类大白话场景描述，当前实验中更推荐使用 BM25 检索 Qwen3.5-VL 生成的 caption。严格去重重跑后，BM25 在这两类场景的 Top20 唯一候选中召回更多 GT 图片，适合作为第一版检索方案。

Embedding 在“雨天，前车刹车灯亮起”这种多条件组合语义场景上 Top10 更强，但 Top20 时 BM25 和 Embedding 都可以召回全部 12 张 GT。因此，当前结论不是“BM25 全面优于 Embedding”，而是：

```text
通用大白话场景检索：优先 BM25
复杂组合语义场景：Embedding 有补充价值
推荐第一版候选规模：Top20，必要时保留 Embedding 作为复杂语义补充
```

## 1. 素材

本轮素材来自 `RetrInput_hh/`。原始文件路径共 99 个，但其中存在重复图片和跨文件夹重复图片。按 SHA256 文件内容哈希做全局去重后，有效唯一图片内容为 79 张。

| 统计口径 | 数量 | 说明 |
|---|---:|---|
| 原始文件条目 | 99 | 按文件路径统计 |
| 去重后唯一图片 | 79 | 按 SHA256 内容哈希全局去重 |

原始子目录文件条目如下：

| 子目录 | 文件条目数 | 用途 |
|---|---:|---|
| `低光照下，前车强逆光` | 5 | 目标场景 |
| `对向来车` | 56 | 目标场景 |
| `雨天，前车刹车灯亮起` | 12 | 目标场景 |
| `雨天` | 26 | 干扰样本 |
| 合计 | 99 | 未去重文件条目 |

按子目录内部去重后的 GT 数如下：

| GT 场景 | 去重后 GT 数 |
|---|---:|
| `低光照下，前车强逆光` | 5 |
| `对向来车` | 53 |
| `雨天，前车刹车灯亮起` | 12 |

注意：同一张图片可能同时出现在多个子目录中，因此各子目录的去重数量相加不等于全局 79 张。

Caption 模型为 Qwen3.5-VL-4B。99 个原始文件条目均成功生成 caption，无失败。

| 阶段 | 模型 | 结果 |
|---|---|---|
| Caption 生成 | Qwen3.5-VL-4B | 99/99 成功 |

## 2. 检索方式

本实验只比较两种方案，不再使用融合检索：

| 方法 | 说明 |
|---|---|
| BM25 | 对 Qwen3.5-VL 生成的 caption 建立词法索引，按 BM25 分数排序 |
| Embedding | 使用 Qwen3-VL-Embedding 对 caption 和 query 编码，再用 FAISS 按相似度排序 |

评价方式改为固定候选数：

| 指标 | 说明 |
|---|---|
| Top10 命中 | 去重后排序前 10 张唯一图片中命中的 GT 数 |
| Top20 命中 | 去重后排序前 20 张唯一图片中命中的 GT 数 |

本报告只统计召回命中，不统计 precision。因此候选里有多少非目标图片，本轮不评价。

## 3. 实验结果

下表为严格重跑后的 Top10 / Top20 命中结果。实验先按 SHA256 内容哈希将 caption 语料去重为 79 条，然后分别用这 79 条语料重新构建 BM25 索引和 Embedding + FAISS 索引。每条 query 都在全量 79 张唯一图片中检索，GT 只用于最后统计命中，不参与检索过滤。

| Query | GT 数 | 方法 | Top10 命中 | Top20 命中 |
|---|---:|---|---:|---:|
| 低光照下，前车强逆光 | 5 | BM25 | 2 | 4 |
| 低光照下，前车强逆光 | 5 | Embedding | 3 | 3 |
| 对向来车 | 53 | BM25 | 10 | 17 |
| 对向来车 | 53 | Embedding | 5 | 13 |
| 雨天，前车刹车灯亮起 | 12 | BM25 | 7 | 12 |
| 雨天，前车刹车灯亮起 | 12 | Embedding | 9 | 12 |

换算成召回率：

| Query | 方法 | Top10 Recall | Top20 Recall |
|---|---|---:|---:|
| 低光照下，前车强逆光 | BM25 | 40.0% | 80.0% |
| 低光照下，前车强逆光 | Embedding | 60.0% | 60.0% |
| 对向来车 | BM25 | 18.9% | 32.1% |
| 对向来车 | Embedding | 9.4% | 24.5% |
| 雨天，前车刹车灯亮起 | BM25 | 58.3% | 100.0% |
| 雨天，前车刹车灯亮起 | Embedding | 75.0% | 100.0% |

## 4. 推理速度

| 阶段 | 方法 | 速度 |
|---|---|---|
| Caption 生成 | Qwen3.5-VL-4B | 14.07 秒/张，99 个文件条目共约 23 分钟 |
| 索引构建 | BM25 | 0.0067 秒 |
| 索引构建 | Embedding + FAISS | 7.7731 秒 |
| 单 query 检索 | BM25 | 0.0006 ~ 0.0010 秒 |
| 单 query 检索 | Embedding + FAISS | 0.0163 ~ 0.0337 秒 |

可以确定：主要耗时来自 caption 生成阶段。检索阶段 BM25 和 Embedding + FAISS 都远快于 caption 生成；其中 BM25 查询耗时最低，且不需要加载 embedding 模型。

## 5. 结果分析

BM25 在“低光照强逆光”和“对向来车”上 Top20 更稳。原因可能是这些 caption 中存在较明确、重复出现的关键词，例如 `low light`、`glare`、`oncoming vehicle`、`opposite lane`、`headlights` 等。对于这种大白话场景描述，BM25 的词项匹配已经能覆盖更多候选。

Embedding 在“雨天，前车刹车灯亮起”上 Top10 更好。这个场景需要同时理解雨天、湿滑路面、前车、红色刹车灯、反光等组合语义，Embedding 在 Top10 命中 9/12，高于 BM25 的 7/12。不过 Top20 时两者都达到 12/12。

“对向来车”的 GT 数较大，去重后仍有 53 张。固定 Top20 本身只允许返回 20 张，因此不可能达到很高的总体召回率。这里更合理的解读是：BM25 的前 20 个候选比 Embedding 更集中地命中了对向来车样本，但 Top20 并不能覆盖全部对向来车场景。

## 6. 初步结论

当前实验建议第一版采用：

```text
Qwen3.5-VL caption -> BM25 -> Top20 候选
```

理由如下：

1. BM25 实现简单，不需要每次查询加载 embedding 模型。
2. BM25 在 2/3 个目标场景上 Top20 命中更多，另一个目标场景与 Embedding 持平。
3. 对大白话式场景 query，caption 中的关键词信号足够强。
4. Top20 候选规模较小，便于后续人工复核或二阶段筛选。

但需要保留一个判断：如果目标场景本身是多条件组合语义，例如“雨天 + 前车 + 刹车灯亮起”，Embedding 更有优势。后续如果要提升复杂场景召回，可以考虑 BM25 与 Embedding 分开保留结果，而不是简单只用一个方法。

## 7. 局限性

- 本轮数据规模较小，去重后只有 79 张唯一图片。
- 当前 GT 主要来自文件夹分类，不是逐图多标签标注。
- 同一图片可能出现在多个子目录，因此严格单标签评价会有偏差。
- 本报告只看召回命中，不看 precision，也不评价误召回成本。
- 检索对象是 caption 文本，不是直接检索原始图片视觉特征。
- Caption 质量会直接影响 BM25 和 Embedding 的最终表现。

## 8. 相关文件

| 类型 | 路径 |
|---|---|
| Caption JSON | `output/reports/caption_retrinput_hh_qwen35.json` |
| Caption JSONL | `output/reports/caption_retrinput_hh_qwen35.jsonl` |
| 严格去重召回结果 JSON | `output/reports/retrinput_hh_strict_dedup_embedding_vs_bm25_topk_recall.json` |
| 严格去重召回结果 CSV | `output/reports/retrinput_hh_strict_dedup_embedding_vs_bm25_topk_recall.csv` |
| 严格去重评估脚本 | `src/experiments/strict_dedup_recall_experiment.py` |
| 原始 99 条百分比召回结果 JSON | `output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.json` |
| 原始 99 条百分比召回结果 CSV | `output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.csv` |
| Query 文件 | `input/processed/scene_queries_specific_en.json` |
| 实验记录 | `docs/dev/experiment_run_record.md` |
