# Scene Retrieval Demo

This folder is a compact demo for the RetrInput_hh caption-based scene retrieval experiment.

It covers:

- image caption generation with Qwen3.5-VL,
- BM25 retrieval over generated captions,
- Qwen3-VL-Embedding + FAISS retrieval over generated captions,
- strict SHA256 image-content deduplication,
- Top10 / Top20 / Top30 recall evaluation.

The demo does not include model weights or image files. Paths are configured in `config/demo_config.json`.

## Files

| File | Purpose |
|---|---|
| `scene_retrieval_demo.py` | Single-file runnable demo |
| `config/demo_config.json` | Model paths, data paths, output paths, Top-k settings |
| `config/caption_prompt.json` | Caption system prompt, user prompt, few-shot examples |
| `config/scene_queries_specific_en.json` | Three scene retrieval queries |
| `CODE_INVENTORY.md` | Mapping from demo functions to original project files |
| `requirements-demo.txt` | Key runtime dependencies |

## Config

Edit `config/demo_config.json` if your local paths differ.

Important fields:

```json
{
  "caption": {
    "model_path": "/mnt/d/qwen-vl/models_cache_qwen3.5/Qwen/Qwen3.5-4B",
    "image_root": "RetrInput_hh"
  },
  "embedding": {
    "embedder_script_path": "/home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py",
    "embedding_model_path": "/home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B"
  }
}
```

## 1. Generate Captions

```bash
PYTHONPATH=src python demo/scene_retrieval_demo/scene_retrieval_demo.py \
  --config demo/scene_retrieval_demo/config/demo_config.json \
  caption
```

Useful smoke test:

```bash
PYTHONPATH=src python demo/scene_retrieval_demo/scene_retrieval_demo.py \
  --config demo/scene_retrieval_demo/config/demo_config.json \
  caption \
  --max-images 1 \
  --output-json output/reports/demo_caption_smoke.json \
  --output-jsonl output/reports/demo_caption_smoke.jsonl
```

## 2. Run BM25 Retrieval

```bash
PYTHONPATH=src python demo/scene_retrieval_demo/scene_retrieval_demo.py \
  --config demo/scene_retrieval_demo/config/demo_config.json \
  retrieve \
  --backends bm25 \
  --top-k 30 \
  --output-json output/reports/demo_bm25_retrieval.json
```

## 3. Run Embedding Retrieval

Requires local Qwen3-VL-Embedding code and model files configured in `demo_config.json`.

```bash
PYTHONPATH=src python demo/scene_retrieval_demo/scene_retrieval_demo.py \
  --config demo/scene_retrieval_demo/config/demo_config.json \
  retrieve \
  --backends qwen3vl_embedding_faiss \
  --top-k 30 \
  --output-json output/reports/demo_embedding_retrieval.json
```

## 4. Run Strict Deduplicated Evaluation

BM25 only:

```bash
PYTHONPATH=src python demo/scene_retrieval_demo/scene_retrieval_demo.py \
  --config demo/scene_retrieval_demo/config/demo_config.json \
  eval \
  --backends bm25
```

BM25 and embedding:

```bash
PYTHONPATH=src python demo/scene_retrieval_demo/scene_retrieval_demo.py \
  --config demo/scene_retrieval_demo/config/demo_config.json \
  eval \
  --backends bm25,qwen3vl_embedding_faiss \
  --top-ks 10,20,30
```

Evaluation semantics:

- Caption records are loaded from `caption_result_json`.
- Images are deduplicated by SHA256 file content.
- BM25 and embedding indexes are rebuilt over the deduplicated corpus.
- Every query searches all deduplicated images.
- GT labels are used only for final recall calculation, not for retrieval filtering.

## Current Experiment Result

Strict deduplicated experiment over 79 unique images:

| Query | GT | BM25 Top10/20/30 | Embedding Top10/20/30 |
|---|---:|---:|---:|
| low-light backlighting | 5 | 2 / 4 / 5 | 3 / 3 / 4 |
| oncoming vehicle | 53 | 10 / 17 / 25 | 5 / 13 / 21 |
| rainy brake lights | 12 | 7 / 12 / 12 | 9 / 12 / 12 |

Recommended first-pass setup:

```text
Qwen3.5-VL caption -> BM25 -> Top20 candidates
```

Use Top30 when higher recall is more important than candidate review cost.
