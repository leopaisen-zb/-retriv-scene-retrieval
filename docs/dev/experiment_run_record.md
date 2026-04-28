# 实验执行记录（issue_* 英文 Query 检索）

### 实验背景

- 目标：在 `issue_*` 数据集上验证「中文仅展示、英文作为检索输入」的检索效果。
- 约束：关闭英文扩展生成，避免非结构化输出引入噪声；严格使用人工指定英文 query。
- 数据规模：`6` 个 issue，共 `120` 张图片（每个 issue `20` 张）。

### 实验配置

- 维护者：`zhuangbie@qq.com`
- 运行环境：`conda activate qwen35vl`
- 图片描述输入：`Retriv/input/raw/issue_*`
- 描述结果：`Retriv/output/reports/caption_issues_all_qwen35.json`
- 检索后端：`qwen3vl_embedding_faiss`
- 检索策略：`--no-expand-english-queries`（关闭扩展），英文 query 直检
- 关键参数：`top_k=20`，`min_score=0.2`
- 查询文件：`Retriv/input/processed/scene_queries.json`（中文展示 + 英文检索）
- 输出文件：
  - `Retriv/output/reports/retrieval_issues_all_qwen3vl_embedding.json`
  - `Retriv/output/reports/retrieval_issues_all_qwen3vl_embedding_recall_summary.csv`
  - `Retriv/output/retrieval_output/<query>/`

### 实验结果

- 总图片数：`120`
- 任务执行成功，`3` 条 query 均完成检索与结果落盘。
- 每条 query 返回 `20` 张（总计 `60` 条召回）。
- 各 query 分数区间（Top1 ~ Top20）：
  - 低光照下，前车强逆光：`0.3977 ~ 0.3114`
  - 对向来车：`0.4230 ~ 0.3766`
  - 雨天，前车刹车灯亮起：`0.3726 ~ 0.3171`
- 按 issue 的召回分布（每条 query 的 20 张）：
  - 低光照下，前车强逆光：`issue_1006000(7)`、`issue_1001000(6)`、`issue_1002000(4)`、其余 `3`
  - 对向来车：`issue_1002000(5)`、`issue_1001000(4)`、`issue_1003000(4)`、`issue_1006000(4)`、其余 `3`
  - 雨天，前车刹车灯亮起：`issue_1006000(11)`、`issue_1005000(4)`、`issue_1001000(2)`、`issue_1002000(2)`、`issue_1004000(1)`

### 实验结论

- 当前配置下，英文直检流程稳定可复现，输出结构完整。
- `对向来车` 的整体匹配分最高，区分度相对更好。
- `雨天，前车刹车灯亮起` 的结果对 `issue_1006000` 聚集明显，说明该 issue 中相关场景特征更集中。
- 三条 query 结果仍存在一定重合，后续可继续优化英文 query 表达以提升场景区分性。

## 2026-04-23 Qwen3-VL-Embedding-2B 文搜文可行性实验（specific English query）

### 实验背景

- 目标：验证更具体的英文场景句子是否能提升 query 到 caption 文本的语义召回可读性。
- 检索范式：文搜文。即 query text 检索 caption text，再按 caption 对应的 `image_path` 将图片复制到 query 目录供人工查看。
- 本次不做图向量检索，也不做 rerank。
- 数据现状：WSL 迁移后的本地语料为 `106` 张图片与 `106` 条成功 caption。

### 实验配置

- 查询文件：`input/processed/scene_queries_specific_en.json`
- Caption 语料：`output/reports/caption_issues_all_qwen35.json`
- Embedding 官方代码：`.external/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py`
- Embedding 模型：`.models/Qwen3-VL-Embedding-2B`
- 后端：`qwen3vl_embedding_faiss`
- 参数：`top_k=20`，未设置 `min_score`
- 输出 JSON：`output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json`
- 输出 CSV：`output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold_recall_summary.csv`
- 图片目录：`output/retrieval_output_specific_en_no_threshold/`

### 实验结果

- 总 query 数：`3`
- 每条 query 返回：`20` 张
- 图片复制结果：`3` 个 query 目录均成功复制 `20` 张图

| query | Recall Count | Score Range | Issue Distribution |
|---|---:|---|---|
| 低光照下，前车强逆光 | 20 | 0.3934 ~ 0.5022 | issue_1003000: 7, issue_1004000: 6, issue_1002000: 3, issue_1006000: 3, issue_1001000: 1 |
| 对向来车 | 20 | 0.4552 ~ 0.5414 | issue_1001000: 5, issue_1006000: 5, issue_1002000: 4, issue_1003000: 3, issue_1004000: 3 |
| 雨天，前车刹车灯亮起 | 20 | 0.3958 ~ 0.5017 | issue_1006000: 12, issue_1002000: 3, issue_1003000: 3, issue_1004000: 1, issue_1005000: 1 |

### 初步结论

- 文搜文链路已跑通：Qwen3-VL-Embedding-2B 能对 caption 文本建向量索引，并按 query 输出可追溯结果。
- 图片按 query 分类复制已验证成功，适合人工快速查看 TopK 结果。
- `雨天，前车刹车灯亮起` 对 `issue_1006000` 聚集明显，说明具体英文 query 对该类场景有较强区分信号。
- `低光照下，前车强逆光` 与 `对向来车` 仍有结果重合，后续需要结合人工查看结果继续优化 query 描述，或引入多条 query 取并/交集策略。

## 2026-04-28 RetrInput_hh Embedding vs BM25 百分比召回实验

### 实验背景

- 目标：在人工分类后的 `RetrInput_hh` 图片集上，对比向量检索与 BM25 的召回能力。
- Query：沿用 `input/processed/scene_queries_specific_en.json`，不新增 query。
- 检索方法：只比较 `qwen3vl_embedding_faiss` 与 `bm25`，不再使用融合方案。
- 指标：按返回比例计算 Recall，不关注 precision/accuracy。
- 返回比例：top `10% / 20% / 30% / 50%`，99 张语料对应候选数 `10 / 20 / 30 / 50`。

### 数据与 GT

- 输入图片根目录：`RetrInput_hh`
- 总图片数：`99`
- Caption 输出：
  - `output/reports/caption_retrinput_hh_qwen35.json`
  - `output/reports/caption_retrinput_hh_qwen35.jsonl`
- Caption 模型路径：`/mnt/d/qwen-vl/models_cache_qwen3.5/Qwen/Qwen3.5-4B`
- Caption 成功数：`99/99`
- GT 规则：使用人工子目录作为严格 GT。

| query | GT 目录 | GT 数 |
|---|---|---:|
| vehicle ahead under strong backlighting in low-light conditions | `低光照下，前车强逆光` | 5 |
| oncoming vehicle approaching in the opposite lane | `对向来车` | 56 |
| rainy road scene with the brake lights of the vehicle ahead illuminated | `雨天，前车刹车灯亮起` | 12 |

说明：`雨天` 目录作为干扰/背景样本；雨天刹车灯 query 只使用 `雨天，前车刹车灯亮起` 的 12 张作为严格 GT。

### 输出文件

- 评估 JSON：`output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.json`
- 评估 CSV：`output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.csv`
- 评估脚本：`src/experiments/retrieval_percentile_recall_experiment.py`
- 设计文档：`docs/superpowers/specs/2026-04-28-retrinput-hh-recall-design.md`
- 实施计划：`docs/superpowers/plans/2026-04-28-retrinput-hh-recall.md`

### 召回结果

| query | method | top10% | top20% | top30% | top50% |
|---|---|---:|---:|---:|---:|
| low-light backlighting | embedding | 1/5 = 0.20 | 3/5 = 0.60 | 3/5 = 0.60 | 4/5 = 0.80 |
| low-light backlighting | BM25 | 2/5 = 0.40 | 3/5 = 0.60 | 4/5 = 0.80 | 5/5 = 1.00 |
| oncoming vehicle | embedding | 4/56 = 0.071 | 11/56 = 0.196 | 17/56 = 0.304 | 32/56 = 0.571 |
| oncoming vehicle | BM25 | 9/56 = 0.161 | 15/56 = 0.268 | 22/56 = 0.393 | 36/56 = 0.643 |
| rainy brake lights | embedding | 5/12 = 0.417 | 10/12 = 0.833 | 12/12 = 1.00 | 12/12 = 1.00 |
| rainy brake lights | BM25 | 5/12 = 0.417 | 8/12 = 0.667 | 11/12 = 0.917 | 12/12 = 1.00 |

### 初步结论

- BM25 在 `低光照下，前车强逆光` 和 `对向来车` 上的召回高于 embedding。
- Embedding 在 `雨天，前车刹车灯亮起` 上更早达到高召回：top20% 已召回 `10/12`，top30% 达到 `12/12`。
- 当 top50% 放宽到 50 张候选时，两种方法在雨天刹车灯 query 上都达到 `12/12`。
- 本实验不解释 precision；返回集合里可能包含大量非 GT 图，后续如需控制误召回，需要补充 precision 或人工误召回分析。

## 2026-04-28 RetrInput_hh 严格去重 Top-k 召回实验

### 实验背景

- 目标：修正上一版 99 条文件路径口径，按图片内容 SHA256 全局去重后重新评估 BM25 与 Embedding。
- 重要口径：检索阶段使用全量 `79` 张唯一图片，不按 GT 目录过滤；GT 只用于最后按 hash 统计命中。
- Query：沿用 `input/processed/scene_queries_specific_en.json`。
- 检索方法：`bm25` vs `qwen3vl_embedding_faiss`。
- 指标：固定 `Top10 / Top20 / Top30` 候选的 hash 级 recall。

### 数据与输出

- 原始 caption 条目：`99`
- 去重后唯一图片语料：`79`
- Caption 输入：`output/reports/caption_retrinput_hh_qwen35.json`
- 评估 JSON：`output/reports/retrinput_hh_strict_dedup_embedding_vs_bm25_topk_recall.json`
- 评估 CSV：`output/reports/retrinput_hh_strict_dedup_embedding_vs_bm25_topk_recall.csv`
- 评估脚本：`src/experiments/strict_dedup_recall_experiment.py`

| query | 去重后 GT 数 |
|---|---:|
| 低光照下，前车强逆光 | 5 |
| 对向来车 | 53 |
| 雨天，前车刹车灯亮起 | 12 |

### 召回结果

| query | method | Top10 | Top20 | Top30 |
|---|---|---:|---:|---:|
| low-light backlighting | BM25 | 2/5 = 40.0% | 4/5 = 80.0% | 5/5 = 100.0% |
| low-light backlighting | Embedding | 3/5 = 60.0% | 3/5 = 60.0% | 4/5 = 80.0% |
| oncoming vehicle | BM25 | 10/53 = 18.9% | 17/53 = 32.1% | 25/53 = 47.2% |
| oncoming vehicle | Embedding | 5/53 = 9.4% | 13/53 = 24.5% | 21/53 = 39.6% |
| rainy brake lights | BM25 | 7/12 = 58.3% | 12/12 = 100.0% | 12/12 = 100.0% |
| rainy brake lights | Embedding | 9/12 = 75.0% | 12/12 = 100.0% | 12/12 = 100.0% |

### 速度

- BM25 索引构建：`0.0067s`
- Embedding + FAISS 索引构建：`7.7731s`
- BM25 单 query：`0.0006s ~ 0.0010s`
- Embedding + FAISS 单 query：`0.0163s ~ 0.0337s`

### 初步结论

- BM25 在 `低光照下，前车强逆光` 和 `对向来车` 的 Top20 / Top30 召回高于 Embedding。
- `雨天，前车刹车灯亮起` 在 Top10 时 Embedding 更高，但 Top20 时 BM25 与 Embedding 均达到 `12/12`。
- 第一版推荐方案仍是 `Qwen3.5-VL caption -> BM25 -> Top20`。
- `对向来车` 去重后 GT 为 `53`，Top20 不可能覆盖全部目标图，因此不能表述为“Top20 覆盖全部对向来车”，只能说 BM25 在固定候选规模下召回更多。
