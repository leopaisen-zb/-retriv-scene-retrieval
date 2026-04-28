# 检索基线与评估规范（实现依据）

维护者：zhuangbie@qq.com

## 1. 文档目标

- 统一当前检索实验结论与评估口径。
- 作为后续实现 `Text -> Image` 检索功能的直接依据。
- 固化输入输出、数据路径、评估方法与阶段目标，保证可复现与可追溯。

## 2. 核心结论

- 推荐主方案：`Embedding + FAISS`。
- 原因：在当前实验中，FAISS 方案整体召回率更稳，无零召回 query，表现均衡。
- `Query Match` 在部分 query 上精确率很高（如“收费站”“洗车设备”达到 1.00），但在“地面裂缝”“安全岛”出现召回率为 0 的情况，稳定性不足。
- 两种方案都出现“精确率偏低”的共性问题（多数在 20%~35%），主要由 `top_k=20` 引入较多负例导致，不应直接归因于模型能力。

## 3. 实验范围与数据规模

- 每类抽取图片数：`10`
- 总样本量：`60` 张图
- 评估对象：
  - 方案 A：语义分析文字匹配（Query Match）
  - 方案 B：文字向量召回（Embedding + FAISS）

## 4. 当前评估结果

## 4.1 Query Match（Qwen 自动语义判定）

| query | Recall | Precision |
|---|---:|---:|
| 收费站 | 0.75 | 1.00 |
| 栏杆 | 0.80 | 0.41 |
| 闸机口 | 0.70 | 0.4375 |
| 地面湿润 | 0.90 | 0.4737 |
| 倒影 | 0.40 | 0.50 |
| 地面文字 | 0.50 | 0.50 |
| 地面箭头 | 0.50 | 0.3571 |
| 洗车设备 | 0.40 | 1.00 |
| 地面裂缝 | 0.00 | N/A |
| 安全岛 | 0.00 | 0.00 |

结论：第一轮效果整体中等，query 间差异明显；“地面湿润”召回率最高，“收费站”“洗车设备”精确率最高；“地面裂缝”“安全岛”检索能力较弱。

## 4.2 Embedding + FAISS（向量召回）

| query | Recall | Precision |
|---|---:|---:|
| 地面湿润 | 90% | 45% |
| 栏杆 | 90% | 45% |
| 收费站 | 80% | 94% |
| 地面文字 | 60% | 75% |
| 地面箭头 | 60% | 40% |
| 闸机口 | 60% | 38% |
| 倒影 | 50% | 42% |
| 安全岛 | 40% | 44% |
| 洗车设备 | 30% | 100% |
| 地面裂缝 | 0% | N/A |

结论：相较 Query Match，FAISS 在召回率上更稳定；除“地面裂缝”外，其他 query 均有有效召回。

## 5. 实验路径与核心文件

- 图片描述脚本：`tests/qwen35/run_qwen35_batch_infer.py`
- 图片描述输入：`tests/qwen35/qwen35_INPUT`
- 语义检索服务路由：`src/dc_ai/bento_qwen3vl_service.py`
- 语义检索实现：`src/dc_ai/services/qwen35_vlm_service.py`
- 语义检索配置：`tests/qwen35/config/query_match_prompt.json`
- 语义检索执行：`tests/qwen35/test_query_match.py`
- 评估脚本：`tests/qwen35/eval_query_match_metrics.py`

## 6. 分阶段流程定义

## 6.1 阶段一：图片描述生成

目标：批量生成每张图片的描述 JSON，作为检索输入语料。

流程：
1. 执行批量推理脚本；
2. 自动读取输入目录图片；
3. 输出每图一个 JSON 到目标目录。

关键输入输出：
- 输入：`tests/qwen35/qwen35_INPUT`
- 输出：`tests/qwen35/output/lab5`

## 6.2 阶段二：Query Match 检索

目标：判断描述文本是否匹配 query，并输出匹配证据。

接口：
- 路径：`/qwen35_vlm/query_match`
- 输入：
  - `description`（字符串）
  - `query`（字符串）
- 输出：
  - `query`
  - `matched`（bool）
  - `evidence`（字符串）

配置文件：`tests/qwen35/config/query_match_prompt.json`
- `system_prompt`
- `user_prompt_template`
- `queries`
- `input_dir`
- `output_dir`

## 6.3 阶段三：Embedding + FAISS 检索

目标：基于描述 embedding 做向量检索与排序。

策略：
- 索引语料：`tests/qwen35/output/lab5` 下 `issue_100*` 前缀，共 60 条；
- 文本字段：`response.results.0.description`；
- 每个 query 配置中文 label 与多组英文 `search_terms`；
- 同一 label 多路召回结果按图片去重，保留最高分再全局排序。

K 值：
- FAISS 检索阶段：`faiss_search_k=30`
- 最终输出截断：`top_k=20`

## 7. 评估真值定义

说明：两套方案使用同一套规则生成真值，非人工逐图标注。

规则：
- 从文件名提取 `issue_id`（示例：`issue_1003000_xxx.json -> 1003000`）；
- 通过 `ISSUE_QUERY_MAP` 判断该 query 是否属于该 issue 标签集。

映射：
- `1001000` -> 栏杆、安全岛
- `1002000` -> 地面箭头、地面文字、地面裂缝
- `1003000` -> 收费站、栏杆
- `1004000` -> 收费站、闸机口
- `1005000` -> 洗车设备
- `1006000` -> 地面湿润、倒影

补充：
- “栏杆”同时出现在 `1001000` 与 `1003000`，在 60 张语料中正例约 20 张。
- “收费站”同理，正例规模也约 20 张。

## 8. 作为后续实现的落地要求

- 主检索链路默认实现为：`Embedding + FAISS`。
- `Query Match` 作为补充策略使用，优先用于高精度场景复核。
- 输出结果必须包含：
  - 命中图片 ID/路径
  - 相似度分数
  - 对应描述文本（用于审计）
  - query 与实验批次标识
- 评估必须同时输出 `Recall` 与 `Precision`，并单独统计“零召回 query”数量。

## 9. 已知风险与改进方向

- 风险 1：`top_k=20` 导致精确率偏低，负例混入显著。
- 风险 2：query 表达不均衡导致某些类别召回不稳定（如“地面裂缝”）。
- 改进方向：
  1. 增加 rerank 阶段（交叉编码器或规则重排）；
  2. 针对弱类 query 增广同义词与英文检索词；
  3. 对 `faiss_search_k` 与 `top_k` 做网格搜索，优化召回-精确率平衡。

## 10. 待确认项

- 当前“向量召回”表格是否已为最终版结果（特别是“地面裂缝=0%”是否在新语料上仍成立）。
- Query Match 与 FAISS 的评估脚本是否完全统一了分母口径（样本总量、正例总量、去重策略）。
