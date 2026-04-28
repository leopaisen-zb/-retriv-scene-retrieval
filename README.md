# Retriv Scene Retrieval

Dashcam scene retrieval experiments comparing caption-based semantic embedding retrieval with BM25 lexical retrieval.

This repository contains a small, reproducible experiment workspace for:

- generating image captions from dashcam frames,
- retrieving images from text scene queries through caption text,
- comparing Qwen embedding retrieval against BM25,
- recording recall-oriented experiment results.

## Experiment Summary

The latest experiment uses `RetrInput_hh/`, a manually grouped 99-image dataset.

Human label folders are used as strict ground truth:

| Query | GT Folder | GT Count |
|---|---|---:|
| vehicle ahead under strong backlighting in low-light conditions | `低光照下，前车强逆光` | 5 |
| oncoming vehicle approaching in the opposite lane | `对向来车` | 56 |
| rainy road scene with the brake lights of the vehicle ahead illuminated | `雨天，前车刹车灯亮起` | 12 |

The `雨天` folder is kept as distractor/background data for the strict rainy-brake-light query.

## Latest Results

Metric: top-percentage recall over all 99 images. Candidate counts are `ceil(99 * percent / 100)`, so top `10% / 20% / 30% / 50%` correspond to `10 / 20 / 30 / 50` returned candidates.

| Query | Method | Top 10% | Top 20% | Top 30% | Top 50% |
|---|---|---:|---:|---:|---:|
| low-light backlighting | Embedding | 1/5 = 20.0% | 3/5 = 60.0% | 3/5 = 60.0% | 4/5 = 80.0% |
| low-light backlighting | BM25 | 2/5 = 40.0% | 3/5 = 60.0% | 4/5 = 80.0% | 5/5 = 100.0% |
| oncoming vehicle | Embedding | 4/56 = 7.1% | 11/56 = 19.6% | 17/56 = 30.4% | 32/56 = 57.1% |
| oncoming vehicle | BM25 | 9/56 = 16.1% | 15/56 = 26.8% | 22/56 = 39.3% | 36/56 = 64.3% |
| rainy brake lights | Embedding | 5/12 = 41.7% | 10/12 = 83.3% | 12/12 = 100.0% | 12/12 = 100.0% |
| rainy brake lights | BM25 | 5/12 = 41.7% | 8/12 = 66.7% | 11/12 = 91.7% | 12/12 = 100.0% |

High-level observations:

- BM25 has higher recall on the low-light and oncoming-vehicle queries.
- Embedding retrieval reaches full recall earlier on the rainy brake-light query.
- This experiment evaluates recall only. It does not evaluate precision or false positives.

## Repository Layout

```text
Retriv/
├── RetrInput_hh/          # 99-image manually grouped evaluation dataset
├── input/processed/       # Query files
├── output/reports/        # Caption and retrieval reports
├── src/
│   ├── experiments/       # CLI experiment entrypoints
│   ├── pipelines/         # Caption and retrieval backends
│   └── utils/             # Shared helpers
├── tests/                 # Pytest coverage for retrieval logic
└── docs/                  # Experiment records, specs, and plans
```

Large local assets are intentionally excluded from git:

- model weights under `.models/`,
- external vendor code under `.external/`,
- local virtual environments,
- raw scratch frames under `input/raw/`,
- downloaded videos under `Downloads/`.

## Key Files

| Path | Purpose |
|---|---|
| `src/experiments/batch_image_to_text_experiment.py` | Batch image caption generation |
| `src/experiments/retrieval_percentile_recall_experiment.py` | Embedding vs BM25 top-percentage recall evaluation |
| `src/pipelines/caption_pipeline.py` | Qwen image captioning pipeline |
| `src/pipelines/retrieval_pipeline.py` | Embedding, BM25, TF-IDF, and hybrid retrieval backends |
| `input/processed/scene_queries_specific_en.json` | Current three scene queries |
| `output/reports/caption_retrinput_hh_qwen35.json` | Captions generated for `RetrInput_hh` |
| `output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.csv` | Compact recall summary |
| `docs/dev/experiment_run_record.md` | Full experiment log |

## Reproducing The Latest Experiment

Set `PYTHONPATH=src` when running scripts.

Generate captions:

```bash
PYTHONPATH=src python src/experiments/batch_image_to_text_experiment.py \
  --model-path /path/to/Qwen3.5-4B \
  --image-root RetrInput_hh \
  --max-new-tokens 512 \
  --output-json output/reports/caption_retrinput_hh_qwen35.json \
  --output-jsonl output/reports/caption_retrinput_hh_qwen35.jsonl
```

Run recall evaluation:

```bash
PYTHONPATH=src python src/experiments/retrieval_percentile_recall_experiment.py \
  --caption-result-json output/reports/caption_retrinput_hh_qwen35.json \
  --query-file input/processed/scene_queries_specific_en.json \
  --label-root RetrInput_hh \
  --output-json output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.json \
  --output-csv output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.csv
```

Run focused tests:

```bash
PYTHONPATH=src python -m pytest -q tests/test_retrieval_percentile_recall_experiment.py
```

## Notes

The embedding backend expects the Qwen3-VL-Embedding implementation and model files to be available locally. They are not committed to this repository because they are large external assets.

