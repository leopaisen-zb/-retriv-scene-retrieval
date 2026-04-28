# Qwen3-VL-Embedding Text Retrieval Feasibility Design

Date: 2026-04-23
Workspace: `/home/leo494/projects/Retriv`

## Goal

Use the official Qwen3-VL-Embedding implementation to run a first feasibility experiment for text-to-text retrieval: user scene query text retrieves image caption text, and the retrieved caption rows map back to source images for manual review.

This experiment answers one question: can Qwen3-VL-Embedding produce useful semantic ranking over the existing 6-issue, 120-image caption corpus?

## Scope

In scope:
- Download or clone the official `QwenLM/Qwen3-VL-Embedding` implementation.
- Download `Qwen/Qwen3-VL-Embedding-2B` weights to a stable local model path.
- Adapt Retriv's existing `Qwen3VLEmbeddingFaissBackend` to load the official `Qwen3VLEmbedder`.
- Keep the existing FAISS retrieval pipeline and output formats.
- Run retrieval over caption text generated from the 6 issue dataset.
- Produce JSON, CSV, and per-query image folders for review.
- Write an experiment record with commands, corpus size, score ranges, issue distribution, and observed feasibility.

Out of scope for this first experiment:
- Image embedding or image-to-text mixed retrieval.
- Qwen3-VL-Reranker.
- Production service packaging.
- Long-lived vector index persistence.
- New manual labels or formal Recall/Precision if ground truth is unavailable.

## Official Source Assumptions

The official GitHub repository is `https://github.com/QwenLM/Qwen3-VL-Embedding`.

The official README describes:
- Qwen3-VL-Embedding as the retrieval model and Qwen3-VL-Reranker as a later ranking model.
- Support for text, image, video, and mixed modal inputs.
- Installation from the repository with `pip install -e .`.
- Model download examples using `huggingface-cli download`.
- A direct Python path style: `from src.models.qwen3_vl_embedding import Qwen3VLEmbedder`.
- Text input shape like `{"text": "...", "instruction": "..."}`.
- Output embeddings produced by `model.process(inputs, normalize=True)`.

The model page for `Qwen/Qwen3-VL-Embedding-2B` describes a 2B embedding model with multilingual support, long context, and configurable embedding output dimension. Retriv will use the official embedder class first, not a SentenceTransformer wrapper, so later image retrieval remains a natural extension.

## Architecture

Retriv already has the right high-level shape:

```text
caption JSON -> load_caption_records -> Qwen3VLEmbeddingFaissBackend -> RetrievalPipeline
             -> JSON/CSV/per-query image folders
```

The implementation should keep this shape. The only intended changes are:

1. Make the Qwen3-VL-Embedding loader compatible with the official repository module layout.
2. Make model and official-repo paths explicit in CLI defaults and error messages.
3. Add a small environment/model smoke check before running the full experiment.
4. Add or update docs so the exact experiment can be repeated.

## Data Flow

1. Ensure the target corpus contains 6 issue folders and 120 sampled images.
2. Ensure caption JSON contains 120 successful rows. If current caption JSON has fewer rows, regenerate or repair the caption corpus before retrieval.
3. Load caption rows from `output/reports/caption_issues_all_qwen35.json`.
4. Encode all caption texts through Qwen3-VL-Embedding with the retrieval instruction.
5. Build `faiss.IndexFlatIP` from normalized caption vectors.
6. Parse query rows from `input/processed/scene_queries.json`.
7. In this first experiment, run with `--no-expand-english-queries` and manually authored English query text.
8. Encode each query through the same embedder.
9. Search FAISS, apply optional `--min-score`, truncate to `top_k`, and write outputs.

## Paths

Recommended WSL-local project paths:

```text
/home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding
/home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B
```

Retriv CLI should still allow overriding both:
- `--embedder-script-path`
- `--embedding-model-path`

If the official repository uses a module file rather than the previous `scripts.py`, `--embedder-script-path` should point to that Python file, and Retriv should load `Qwen3VLEmbedder` from it.

The WSL execution environment should use:

```text
/home/leo494/miniforge3/envs/dc25/bin/python
```

## Experiment Query Set

Use `input/processed/scene_queries.json`:

```json
{
  "queries": [
    {
      "query": "低光照下，前车强逆光",
      "english_query": "low-light scene with strong backlight from the vehicle ahead"
    },
    {
      "query": "对向来车",
      "english_query": "oncoming vehicle in opposite lane"
    },
    {
      "query": "雨天，前车刹车灯亮起",
      "english_query": "rainy road with brake lights on from the vehicle ahead"
    }
  ]
}
```

The Chinese text is display-only in this first experiment. The English text is the retrieval input.

## Output Contract

Keep existing output shape:

- JSON: `output/reports/retrieval_issues_all_qwen3vl_embedding.json`
- CSV: `output/reports/retrieval_issues_all_qwen3vl_embedding_recall_summary.csv`
- Per-query folders: `output/retrieval_output/<query>/`
- Per-query folder manifest: `manifest.json`

The JSON summary must include:
- backend name
- caption result JSON path
- corpus image count
- total queries
- top_k
- min_score
- whether English expansion was enabled
- model path
- embedder source path

## Feasibility Criteria

The first experiment is considered feasible if:

- The model loads locally without requiring code edits outside Retriv and the official Qwen repo.
- 120 caption records can be encoded without runtime failure.
- All 3 queries return ranked results.
- JSON, CSV, and per-query image folders are written.
- At least one query shows a meaningful concentration in expected issue folders, or score ranking is interpretable enough for manual review.

This is not a final retrieval-quality claim. It is a smoke-quality experiment proving that Qwen3-VL-Embedding can drive the existing search workflow.

## Error Handling

User-facing errors should stay Chinese, matching existing Retriv code.

Important failure cases:
- Official repository path missing.
- Official embedder Python file missing or does not expose `Qwen3VLEmbedder`.
- Model weights path missing.
- Required dependencies missing or too old.
- Caption corpus has fewer than the expected 120 successful rows.
- CUDA/model load failure.

## Review Checklist

Before accepting the experiment result:

- Confirm actual caption corpus count is 120.
- Confirm the run used `Qwen/Qwen3-VL-Embedding-2B`, not TF-IDF.
- Confirm the run used `--no-expand-english-queries`.
- Confirm each query has 20 rows when `top_k=20` and `min_score=0.2`, unless the score threshold filters rows.
- Inspect issue distribution per query.
- Record exact commands and output paths in `docs/dev/experiment_run_record.md` or a new experiment result doc.
