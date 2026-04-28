# Code Inventory

This demo consolidates the project code paths used for the RetrInput_hh caption and retrieval experiment.

| Demo Function | Source File |
|---|---|
| Image loading | `src/utils/image_loader.py` |
| Caption prompt | `src/utils/prompt_config.py` |
| Single-image caption pipeline | `src/pipelines/caption_pipeline.py` |
| Batch caption command | `src/experiments/batch_image_to_text_experiment.py` |
| Caption record loading | `src/pipelines/retrieval_pipeline.py` |
| BM25 retrieval | `src/pipelines/retrieval_pipeline.py` |
| Qwen3-VL-Embedding + FAISS retrieval | `src/pipelines/retrieval_pipeline.py` |
| Strict hash deduplication and Top-k recall | `src/experiments/strict_dedup_recall_experiment.py` |
| Scene query config | `input/processed/scene_queries_specific_en.json` |

The demo does not copy model weights or input images. Paths are configured in `config/demo_config.json`.
