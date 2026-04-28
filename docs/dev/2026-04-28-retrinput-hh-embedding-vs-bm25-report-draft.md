# RetrInput_hh 场景检索实验报告初稿

日期：2026-04-28

## 1. 实验目的

本实验旨在评估两类基于图片描述文本的检索方法在行车场景检索任务中的召回能力：

- 向量检索：使用 Qwen3-VL-Embedding 对图片描述和查询文本进行向量化，并按相似度排序。
- BM25：使用图片描述文本构建词法检索索引，并按 BM25 分数排序。

本轮实验只关注召回率，不评价准确率、精确率或误召回情况。实验问题可以表述为：

> 在相同候选返回比例下，Embedding 和 BM25 哪个方法能召回更多人工标注的目标场景图片？

## 2. 实验数据

实验输入为 `RetrInput_hh/` 目录下的 99 张图片。该数据集已由人工按场景分入不同子目录，因此本实验将子目录名作为 ground truth 标签。

| 子目录 | 图片数 | 用途 |
|---|---:|---|
| `低光照下，前车强逆光` | 5 | 低光照强逆光 query 的严格 GT |
| `对向来车` | 56 | 对向来车 query 的严格 GT |
| `雨天，前车刹车灯亮起` | 12 | 雨天前车刹车灯 query 的严格 GT |
| `雨天` | 26 | 干扰/背景样本，不作为本轮三条 query 的 GT |

其中，`雨天` 文件夹中的图片可能与其他雨天相关场景存在语义重叠，但本实验采用严格 GT：对于“雨天，前车刹车灯亮起”这一 query，只将 `雨天，前车刹车灯亮起` 文件夹中的 12 张图片视为相关图片。

## 3. 查询设置

实验沿用 `input/processed/scene_queries_specific_en.json` 中的三条英文查询，不新增 query。

| 查询 | GT 目录 | GT 数 |
|---|---|---:|
| `vehicle ahead under strong backlighting in low-light conditions` | `低光照下，前车强逆光` | 5 |
| `oncoming vehicle approaching in the opposite lane` | `对向来车` | 56 |
| `rainy road scene with the brake lights of the vehicle ahead illuminated` | `雨天，前车刹车灯亮起` | 12 |

## 4. 实验流程

### 4.1 图片描述生成

首先使用 Qwen3.5-VL 对 `RetrInput_hh/` 下所有图片生成英文/结构化描述文本，并将结果保存为 JSON/JSONL，供后续检索方法共同使用。

Caption 配置如下：

| 配置项 | 值 |
|---|---|
| 输入目录 | `RetrInput_hh` |
| Caption 模型 | `/mnt/d/qwen-vl/models_cache_qwen3.5/Qwen/Qwen3.5-4B` |
| 最大生成长度 | `512` tokens |
| 输出 JSON | `output/reports/caption_retrinput_hh_qwen35.json` |
| 输出 JSONL | `output/reports/caption_retrinput_hh_qwen35.jsonl` |

Caption 结果：

| 指标 | 数值 |
|---|---:|
| 总图片数 | 99 |
| 成功图片数 | 99 |
| 失败图片数 | 0 |
| 平均单图推理耗时 | 14.07 秒 |

### 4.2 检索方法

本实验比较两种方法：

| 方法 | 说明 |
|---|---|
| `qwen3vl_embedding_faiss` | 对 caption 文本和 query 文本计算 embedding，并使用 FAISS 按相似度排序 |
| `bm25` | 对 caption 文本建立 BM25 词法索引，并按词项匹配分数排序 |

本轮实验不使用融合检索方案。

### 4.3 评价指标

由于用户关注“是否能召回目标图片”，本实验采用 top-percentage recall，而不是固定 Top-k 或绝对分数阈值。

具体做法：

1. 对每个 query，每个方法都对 99 张图片排序。
2. 分别取排序结果前 `10% / 20% / 30% / 50%` 作为候选集合。
3. 99 张图片对应候选数为 `10 / 20 / 30 / 50`。
4. 计算候选集合中命中的 GT 图片数量。

召回率公式：

```text
Recall = recalled_gt_count / gt_count
```

本实验不统计 precision，因此返回候选中有多少非目标图片不在本报告中评价。

## 5. 实验结果

| Query | Method | Top 10% | Top 20% | Top 30% | Top 50% |
|---|---|---:|---:|---:|---:|
| 低光照下，前车强逆光 | Embedding | 1/5 = 20.0% | 3/5 = 60.0% | 3/5 = 60.0% | 4/5 = 80.0% |
| 低光照下，前车强逆光 | BM25 | 2/5 = 40.0% | 3/5 = 60.0% | 4/5 = 80.0% | 5/5 = 100.0% |
| 对向来车 | Embedding | 4/56 = 7.1% | 11/56 = 19.6% | 17/56 = 30.4% | 32/56 = 57.1% |
| 对向来车 | BM25 | 9/56 = 16.1% | 15/56 = 26.8% | 22/56 = 39.3% | 36/56 = 64.3% |
| 雨天，前车刹车灯亮起 | Embedding | 5/12 = 41.7% | 10/12 = 83.3% | 12/12 = 100.0% | 12/12 = 100.0% |
| 雨天，前车刹车灯亮起 | BM25 | 5/12 = 41.7% | 8/12 = 66.7% | 11/12 = 91.7% | 12/12 = 100.0% |

## 6. 结果分析

### 6.1 低光照强逆光场景

在 `低光照下，前车强逆光` query 上，BM25 的整体召回表现优于 embedding。

- Top 10% 时，BM25 召回 2/5，Embedding 召回 1/5。
- Top 30% 时，BM25 召回 4/5，Embedding 召回 3/5。
- Top 50% 时，BM25 达到 5/5 全召回，Embedding 为 4/5。

初步判断，该场景的 caption 中可能包含较明确的词法信号，例如 `low light`、`glare`、`headlights`、`vehicle ahead` 等，因此 BM25 能通过关键词匹配获得较好召回。

### 6.2 对向来车场景

在 `对向来车` query 上，BM25 也优于 embedding。

- Top 10% 时，BM25 召回 9/56，Embedding 召回 4/56。
- Top 30% 时，BM25 召回 22/56，Embedding 召回 17/56。
- Top 50% 时，BM25 召回 36/56，Embedding 召回 32/56。

该类别 GT 数较多，占总语料比例较高。BM25 的优势说明 caption 中关于 `oncoming vehicle`、`opposite lane`、`headlights`、`lane` 等词汇可能较稳定，词法匹配可以覆盖更多同类样本。

### 6.3 雨天前车刹车灯场景

在 `雨天，前车刹车灯亮起` query 上，embedding 表现更好。

- Top 10% 时，两者均召回 5/12。
- Top 20% 时，Embedding 召回 10/12，BM25 召回 8/12。
- Top 30% 时，Embedding 达到 12/12 全召回，BM25 为 11/12。
- Top 50% 时，两者均达到 12/12。

这说明该场景可能包含较强的整体语义组合，例如“雨天、湿滑路面、前车、红色刹车灯、反光”等。Embedding 对组合语义的表达更强，因此在中等候选规模下更早达到高召回。

## 7. 初步结论

本轮实验得到以下初步结论：

1. 在 `低光照下，前车强逆光` 和 `对向来车` 两个场景上，BM25 的召回率高于 embedding。
2. 在 `雨天，前车刹车灯亮起` 场景上，embedding 的召回率高于 BM25，并且更早达到全召回。
3. BM25 在 caption 中关键词稳定、目标词汇明确的场景中表现较好。
4. Embedding 在需要组合多个视觉语义条件的场景中更有优势。
5. 由于本实验只评价 recall，不能据此判断哪个方法整体“更准确”；如果关注误召回，需要进一步评价 precision 或人工分析候选中的非 GT 图片。

## 8. 局限性

本实验仍存在以下限制：

- 数据规模较小，总计 99 张图片。
- GT 来自人工文件夹分类，但不同场景之间可能存在语义重叠。
- `对向来车` 类别数量明显大于其他类别，类别分布不均衡。
- 只评价 recall，没有评价 precision、F1 或误召回成本。
- 检索对象是 caption 文本，不是直接对图片本身做图文向量检索。
- Caption 质量会直接影响 BM25 和 embedding 两种检索方法。

## 9. 后续建议

后续可以从以下方向继续完善：

1. 增加 precision 或人工误召回分析，评估返回候选中的非目标图片比例。
2. 为每个 query 增加更细粒度的人工相关性标注，而不是只依赖文件夹级标签。
3. 对 query 进行多版本改写，观察 BM25 和 embedding 对 query 表述的敏感性。
4. 增加更多场景类别和更多样本，验证结论是否稳定。
5. 尝试 caption prompt 优化，观察更结构化的 caption 是否能提升 BM25 或 embedding 召回。

## 10. 相关文件

| 类型 | 路径 |
|---|---|
| Caption 结果 | `output/reports/caption_retrinput_hh_qwen35.json` |
| Recall JSON | `output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.json` |
| Recall CSV | `output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.csv` |
| 评估脚本 | `src/experiments/retrieval_percentile_recall_experiment.py` |
| 查询文件 | `input/processed/scene_queries_specific_en.json` |
| 实验记录 | `docs/dev/experiment_run_record.md` |

