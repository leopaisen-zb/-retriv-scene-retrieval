# TODO 与计划

维护者：zhuangbie@qq.com

## 待办事项

- [x] 准备 `input/raw/example_data` 初始样本并完成整目录复制。
- [x] 完成批量图像描述脚本 `batch_image_to_text_experiment.py`。
- [x] 完成 Qwen3-VL-8B 与 Qwen3.5-4B 全量 7 张图实验。
- [x] 输出实验结果到 `output/reports/*.json` 与 `*.jsonl`。
- [ ] 增加“描述质量评估”脚本（关键词覆盖率、方位词覆盖率）。
- [ ] 基于 `caption_results_all_qwen35.json` 构建首版检索索引。
- [ ] 实现 `RetrievalBackend` 的可运行版本（先本地轻量实现）。
- [ ] 设计文本检索图片评测集与指标（Recall@K、MRR）。
- [x] 形成检索实现依据文档：`docs/design/retrieval_baseline_and_eval_spec.md`。

## 说明

- 所有实验计划与执行记录集中维护在本文件。
- 涉及流程较复杂时，补充 `mermaid` 图示到 `docs/design/`。
