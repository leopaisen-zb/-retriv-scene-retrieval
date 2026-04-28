# 检索接口规范（草案）

维护者：zhuangbie@qq.com

## 1. 查询输入

- `query`：检索文本，非空字符串。
- `top_k`：返回数量，正整数。
- `model_version`：可选，指定检索使用的模型版本。
- `experiment_tag`：可选，实验批次标识（如 `qwen35_baseline_20260420`）。

## 2. 输出结构

```json
{
  "meta": {
    "model_version": "string",
    "experiment_tag": "string"
  },
  "items": [
    {
      "image_id": "string",
      "image_path": "string",
      "score": 0.0,
      "caption": "string",
      "source_result_file": "string"
    }
  ]
}
```

## 3. 约束

- `score` 越大表示相关性越高。
- `items` 按 `score` 降序返回。
- 失败场景需抛出中文错误信息。
- `caption` 用于调试与人工复核，可为空。
- `source_result_file` 需可追溯到 `output/reports` 的结果文件。

## 4. 当前数据源约定

- 可用描述结果文件：
  - `output/reports/caption_results_all.json`
  - `output/reports/caption_results_all_qwen35.json`
- 检索原型阶段默认读取上述结果并构建轻量倒排或向量索引。
