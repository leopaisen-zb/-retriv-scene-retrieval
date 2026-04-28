# Retriv 运行指南

维护者：zhuangbie@qq.com

## 1. 环境准备

- Python 3.10 及以上。
- Linux 环境。
- 建议在虚拟环境中安装依赖。
- 运行测试前确认 CUDA 可用。

## 2. 建议依赖

```bash
pip install torch transformers pillow qwen-vl-utils rank_bm25
```

说明：如需特定 CUDA 版本，请按实际环境替换 `torch` 安装方式。

## 3. 提示词版本约定

- 当前敲定版本位于：`src/utils/prompt_config.py`
- `SYSTEM_INSTRUCTION` 严格使用 3 点要求：
  1. 主体与空间关系（前后左右）
  2. 光照、氛围与细节
  3. 中性、观察式语气
- 输出限制：`Limit the response within 2048 tokens.`

## 4. Image -> Text 运行方式

在 `Retriv/` 目录执行：

### 4.1 单图：图片路径输入

```bash
PYTHONPATH=src python src/experiments/image_to_text_experiment.py \
  --model-path <你的模型路径或模型名> \
  --image-path input/raw/example.jpg \
  --output-text output/reports/example_caption.txt \
  --output-json output/reports/example_caption.json
```

### 4.2 单图：base64 文件输入

```bash
PYTHONPATH=src python src/experiments/image_to_text_experiment.py \
  --model-path <你的模型路径或模型名> \
  --image-b64-file input/raw/example.b64.txt \
  --output-text output/reports/example_caption.txt \
  --output-json output/reports/example_caption.json
```

### 4.3 批量：目录图片输入

```bash
PYTHONPATH=src python src/experiments/batch_image_to_text_experiment.py \
  --model-path <你的模型路径或模型名> \
  --image-root input/raw/example_data \
  --output-json output/reports/caption_results_all.json \
  --output-jsonl output/reports/caption_results_all.jsonl \
  --max-images 0 \
  --max-new-tokens 2048
```

### 4.4 Qwen3.5 专用运行方式

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate qwen35vl
PYTHONPATH=src python src/experiments/batch_image_to_text_experiment.py \
  --model-path /defaultShare/qwen-vl/models_cache_qwen3.5/Qwen/Qwen3.5-4B \
  --image-root input/raw/example_data \
  --output-json output/reports/caption_results_all_qwen35.json \
  --output-jsonl output/reports/caption_results_all_qwen35.jsonl \
  --max-images 0 \
  --max-new-tokens 2048
```

## 5. Text -> Image 向量召回

当前已支持以下检索后端：

- `qwen3vl_embedding_faiss`：语义向量检索
- `bm25`：BM25 词法检索
- `hybrid_embedding_bm25`：语义检索 + BM25 混合检索
- `tfidf_faiss`：旧的 TF-IDF 基线

### 5.1 使用默认场景 query（推荐）

```bash
PYTHONPATH=src python src/experiments/text_to_image_retrieval_experiment.py \
  --backend qwen3vl_embedding_faiss \
  --embedder-script-path /defaultShare/qwen-vl/Qwen/Qwen3-VL-Embedding/scripts.py \
  --embedding-model-path /defaultShare/qwen-vl/models_cache_qwen3-VL-Embedding-2B/Qwen/Qwen3-VL-Embedding-2B \
  --caption-result-json output/reports/caption_results_all_qwen35.json \
  --top-k 5 \
  --output-json output/reports/retrieval_scene_result.json
```

默认场景 query：
- vehicle ahead under strong backlighting in low-light conditions
- oncoming vehicle approaching in the opposite lane
- rainy road scene with the brake lights of the vehicle ahead illuminated

### 5.1.1 检索输出里如何看「每个 query 召回了哪些图」

指定 `--output-json` 时，JSON 内会包含：

- `query_recall_summary`：数组，每项为一条 query 及扁平列表 `recalled_image_filenames`、`recalled_image_paths`、`recalled_image_ids`、`recall_count`。
- `query_results` 中每条除 `items` 外还有：`recalled_image_paths`、`recalled_image_filenames`、`recalled_image_ids`、`recall_count`。

同时会在同目录自动生成 CSV（便于 Excel 透视）：`{json 文件名去掉扩展名}_recall_summary.csv`。也可用 `--output-csv` 指定 CSV 路径。

默认还会把每个 query 的召回图片**复制**到 `output/retrieval_output/<与 query 一致的目录名>/`（例如 `output/retrieval_output/vehicle ahead under strong backlighting in low-light conditions/`），文件名形如 `01_原图名.jpg`，同目录下有 `manifest.json` 记录来源路径与分数。不需要复制时加 `--no-organize-by-query`；根目录可改 `--retrieval-output-dir`。

### 5.1.2 英文多路 query 合并检索（默认开启）

为提升跨语言与场景表述召回，默认会对**每条用户 query**（可为中文）做：

1. 用 `--query-expand-model-path` 指定的模型生成 `expand-query-count` 条**英文**短语（第 1 条为翻译，其余为相近驾驶场景表述）。
2. 每条英文短语分别做 embedding 检索。
3. 同一图片在多条英文 query 下的相似度取**最大值**，再按该分数排序取 `--top-k`。

关闭该行为（仅单条原始 query 检索）：加 `--no-expand-english-queries`。

相关参数：`--query-expand-model-path`、`--expand-query-count`（默认 4）、`--query-expand-max-new-tokens`。

结果 JSON / CSV / 各子目录 `manifest.json` 中会包含 `expanded_english_queries` 字段便于核对。

### 5.1.3 为什么每个 query 下面都是全部图片（例如都是 7 张）

索引里的**语料只有 7 张**时，向量检索单次最多只能返回 7 条。若 `--top-k` 大于等于 7（例如 10），则对每个 query 都会返回**全部 7 张**，只是排序不同。这不是复制逻辑错误。

处理方式：

- 把 `--top-k` 改成小于语料张数（例如 `--top-k 3`），每个目录里最多 3 张。
- 或加 `--min-score 0.32`（数值需按你数据试），先对全语料打分再丢掉低分，再截断到 `top-k`。
- 根本办法是扩大语料（更多图片描述进索引）。

### 5.2 指定单条 query

```bash
  PYTHONPATH=src python src/experiments/text_to_image_retrieval_experiment.py \
  --backend qwen3vl_embedding_faiss \
  --query "vehicle ahead under strong backlighting in low-light conditions" \
  --caption-result-json output/reports/caption_results_all_qwen35.json \
  --top-k 10 \
  --output-json output/reports/retrieval_single_query.json
```

### 5.3 使用 query 文件批量检索

```bash
PYTHONPATH=src python src/experiments/text_to_image_retrieval_experiment.py \
  --backend qwen3vl_embedding_faiss \
  --query-file input/processed/scene_queries.json \
  --caption-result-json output/reports/caption_results_all_qwen35.json \
  --top-k 10 \
  --output-json output/reports/retrieval_batch_query.json
```

`scene_queries.json` 示例：

```json
{
  "queries": [
    "vehicle ahead under strong backlighting in low-light conditions",
    "oncoming vehicle approaching in the opposite lane",
    "rainy road scene with the brake lights of the vehicle ahead illuminated"
  ]
}
```

## 6. 结果文件说明

- `caption_results_all.json`：Qwen3-VL-8B 全量结果。
- `caption_results_all_qwen35.json`：Qwen3.5-4B 全量结果。
- `caption_results_all*.jsonl`：逐条结果，便于流式处理。
- `docs/dev/experiment_run_record.md`：实验记录主文档。

## 7. 下一步建议

- 接入真实检索后端实现 `RetrievalBackend`。
- 补充描述质量评估指标（关键词覆盖、方位准确率）。
- 基于 JSON 结果构建文本检索样本集。
