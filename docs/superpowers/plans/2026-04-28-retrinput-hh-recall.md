# RetrInput HH Recall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a simple recall-only experiment comparing vector retrieval against BM25 on `RetrInput_hh`.

**Architecture:** Keep caption generation in the existing `batch_image_to_text_experiment.py`. Add a focused evaluation script that loads caption records, runs embedding and BM25 over the full corpus, and computes top-percentage recall from human-labeled directories.

**Tech Stack:** Python, existing `src/pipelines/retrieval_pipeline.py` backends, pytest, Qwen3.5 caption model, Qwen3-VL embedding model.

---

### Task 1: Add Percentile Recall Evaluation Helpers

**Files:**
- Create: `tests/test_retrieval_percentile_recall_experiment.py`
- Create: `src/experiments/retrieval_percentile_recall_experiment.py`

- [ ] **Step 1: Write failing tests**

Create tests that import `percentile_candidate_count` and `evaluate_percentile_recall`. Cover `ceil` cutoff behavior and full-path GT matching.

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src pytest -q tests/test_retrieval_percentile_recall_experiment.py`

Expected: fails because `experiments.retrieval_percentile_recall_experiment` does not exist.

- [ ] **Step 3: Implement minimal helpers**

Create `src/experiments/retrieval_percentile_recall_experiment.py` with:

- `percentile_candidate_count(total_count: int, percent: int) -> int`
- `evaluate_percentile_recall(items, gt_paths, percentiles, total_count) -> list[dict[str, object]]`

Use `math.ceil(total_count * percent / 100)`, clamp to `[1, total_count]`, and compare by resolved image path strings.

- [ ] **Step 4: Run tests and verify pass**

Run: `PYTHONPATH=src pytest -q tests/test_retrieval_percentile_recall_experiment.py`

Expected: all tests pass.

### Task 2: Add CLI Evaluation Script

**Files:**
- Modify: `src/experiments/retrieval_percentile_recall_experiment.py`
- Modify: `tests/test_retrieval_percentile_recall_experiment.py`

- [ ] **Step 1: Add tests for default query-to-label mapping and output row shape**

Test that the default three existing queries map to strict GT folders:

- low-light query -> `低光照下，前车强逆光`
- oncoming query -> `对向来车`
- rainy brake-light query -> `雨天，前车刹车灯亮起`

- [ ] **Step 2: Run tests and verify failure**

Run: `PYTHONPATH=src pytest -q tests/test_retrieval_percentile_recall_experiment.py`

Expected: fails because mapping/output helpers are missing.

- [ ] **Step 3: Implement CLI**

Add arguments:

- `--caption-result-json`
- `--query-file`
- `--label-root`
- `--output-json`
- `--output-csv`
- `--percentiles`, default `10,20,30,50`
- embedding model args matching `text_to_image_retrieval_experiment.py`

The CLI should:

- load caption records
- parse queries using existing `_parse_queries`
- build GT sets from `label-root/<label>`
- run `Qwen3VLEmbeddingFaissBackend` and `BM25RetrievalBackend` with `top_k=corpus_size`
- compute recall summaries for each backend
- write JSON and CSV

- [ ] **Step 4: Run focused tests**

Run: `PYTHONPATH=src pytest -q tests/test_retrieval_percentile_recall_experiment.py`

Expected: all tests pass.

### Task 3: Run Caption Pipeline For RetrInput HH

**Files:**
- Output: `output/reports/caption_retrinput_hh_qwen35.json`
- Output: `output/reports/caption_retrinput_hh_qwen35.jsonl`

- [ ] **Step 1: Run caption command**

Run with the `dc25` Conda environment:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh
conda activate dc25
PYTHONPATH=src python src/experiments/batch_image_to_text_experiment.py \
  --model-path /defaultShare/qwen-vl/models_cache_qwen3.5/Qwen/Qwen3.5-4B \
  --image-root RetrInput_hh \
  --output-json output/reports/caption_retrinput_hh_qwen35.json \
  --output-jsonl output/reports/caption_retrinput_hh_qwen35.jsonl
```

- [ ] **Step 2: Verify caption output**

Check that `summary.total_images == 99`, `summary.success_images == 99`, and `summary.failed_images == 0`.

### Task 4: Run Embedding vs BM25 Percentile Recall Experiment

**Files:**
- Output: `output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.json`
- Output: `output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.csv`

- [ ] **Step 1: Run evaluation command**

Run with the `dc25` Conda environment:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh
conda activate dc25
PYTHONPATH=src python src/experiments/retrieval_percentile_recall_experiment.py \
  --caption-result-json output/reports/caption_retrinput_hh_qwen35.json \
  --query-file input/processed/scene_queries_specific_en.json \
  --label-root RetrInput_hh \
  --output-json output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.json \
  --output-csv output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.csv
```

- [ ] **Step 2: Verify evaluation output**

Check that the JSON has 3 queries, 2 backends, and 4 percent cutoffs per query/backend. Check that no hybrid backend appears.

### Task 5: Record Experiment

**Files:**
- Modify: `docs/dev/experiment_run_record.md`

- [ ] **Step 1: Append a concise run record**

Record:

- input root `RetrInput_hh`
- caption output paths
- evaluation output paths
- GT mapping
- metric definition
- final recall table summary

- [ ] **Step 2: Run tests**

Run: `PYTHONPATH=src pytest -q tests/test_retrieval_percentile_recall_experiment.py`

Expected: all focused tests pass.

