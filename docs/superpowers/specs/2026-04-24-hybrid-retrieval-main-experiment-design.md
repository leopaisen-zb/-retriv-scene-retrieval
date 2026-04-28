# Hybrid Retrieval Main Experiment Design

Date: 2026-04-24
Workspace: `/home/leo494/projects/Retriv`

## Goal

Run one constrained hybrid retrieval experiment whose only question is:

Can `hybrid` perform at least no worse than the frozen `embedding` baseline under manual top-k image review?

This is not a production redesign. It is a bounded experiment that keeps the current semantic baseline intact and limits all changes to a new hybrid branch.

## Non-Negotiable Constraints

The following inputs are frozen and must not be modified:

- embedding retrieval configuration
- existing embedding result files
- caption result JSON
- caption prompt and prompt configuration

Specifically:

- Baseline semantic report is:
  `output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json`
- Frozen caption corpus is:
  `output/reports/caption_issues_all_qwen35.json`

The experiment may create new files and new folders, but must not overwrite or mutate the baseline chain.

## Baseline Definition

The semantic baseline is the existing embedding retrieval run:

- backend: `qwen3vl_embedding_faiss`
- embedding model: `Qwen3-VL-Embedding-2B`
- retrieval corpus: caption text loaded from `caption_issues_all_qwen35.json`
- query source: the query set already reflected in the baseline report
- top_k: `20`
- min_score: `null`
- English expansion: disabled

For this experiment, the baseline semantic ranking is treated as authoritative input. The hybrid branch may read it, compare against it, and derive new outputs from it, but may not reconfigure it or replace it.

## Experiment Scope

In scope:

- Create one BM25-specific lexical view derived from the frozen caption text.
- Run one conservative hybrid ranking experiment.
- Write all outputs to new files and new folders.
- Produce side-by-side comparison artifacts for manual review.

Out of scope:

- Changing caption generation
- Changing caption prompts
- Changing the embedding model
- Changing embedding CLI defaults or baseline config files
- Re-running the baseline to get a different semantic ranking
- Adding new evaluation labels or automated ground-truth metrics

## Architecture

The main experiment branch should have this shape:

```text
frozen caption JSON -> deterministic BM25 lexical-view builder -> BM25 ranking
frozen embedding report --------------------------------------> semantic ranking
semantic ranking + BM25 ranking -> conservative hybrid reranker -> new report + comparison folders
```

Key principle:

`hybrid` is not an equal-weight fusion of two peers. It is an embedding-led reranker where BM25 is only allowed to make limited changes in the tail of the semantic ranking.

## BM25 Lexical View

The BM25 branch must not consume raw long-form caption text directly. Instead, it should use a derived lexical view built from the frozen caption corpus.

Requirements for the lexical view:

- built deterministically from the existing caption text
- no new model inference
- no rewriting of the original caption file
- stored as a separate artifact for inspection

Recommended transformation:

1. Remove obvious retrieval noise:
   timestamps, camera channel markers, plate-like strings, section labels such as `Foreground`, `Background`, `Overall Composition`
2. Keep scene-bearing terms and short phrases:
   weather, wet-road cues, lighting cues, glare/backlighting cues, opposing-traffic cues, lead-vehicle cues, brake-light cues
3. Compress to a short lexical text suitable for BM25

Example:

```text
raw caption:
long descriptive paragraph with layout prose, timestamps, and general scene narration

bm25_text:
wet road lead vehicle brake lights illuminated low visibility reflections
```

The lexical-view artifact should contain, at minimum:

- `image_id`
- `image_path`
- `source_caption`
- `bm25_text`

Optional but recommended for debugging:

- `kept_phrases`
- `dropped_fragments`

## Hybrid Fusion Strategy

This experiment uses one conservative hybrid strategy only.

Rules:

1. Semantic baseline top-5 is frozen.
2. Final result size remains top-20.
3. Final candidate set is restricted to the semantic baseline top-20.
4. BM25 is only allowed to affect ranks 6-20.
5. BM25 may not introduce any image that is outside semantic top-20.

This means the experiment is a tail rerank, not a fresh fusion.

Rationale:

- protects the strongest part of the existing semantic baseline
- isolates whether BM25 can improve tail quality without damaging the head
- keeps manual review simple because all differences are localized

## Data Flow

1. Read the frozen embedding report.
2. Read the frozen caption JSON.
3. Build the BM25 lexical-view artifact from caption text.
4. Run BM25 ranking using the lexical view.
5. For each query:
   keep semantic ranks 1-5 unchanged
6. For each query:
   take semantic ranks 6-20 and allow BM25-based reranking only within that slice
7. Write one new hybrid report and one comparison directory tree

## Output Contract

The experiment must write only new outputs.

Recommended paths:

- BM25 lexical view:
  `output/artifacts/bm25_views/hybrid_main_experiment.json`
- Hybrid report:
  `output/reports/retrieval_issues_all_hybrid_main_experiment.json`
- Hybrid image folders:
  `output/retrieval_output_hybrid_main_experiment/`
- Comparison artifacts:
  `output/comparisons/hybrid_main_experiment/`

Per query, comparison output should include:

- `embedding/`
- `hybrid/`
- `only_embedding/`
- `only_hybrid/`
- `manifest.json`

`manifest.json` should record, at minimum:

- query text
- frozen embedding report path
- hybrid report path
- embedding image ids in rank order
- hybrid image ids in rank order
- intersection image ids
- only-embedding image ids
- only-hybrid image ids

## Manual Review Protocol

Success is judged by human inspection, not automatic metrics.

Review process:

1. For each of the 3 queries, inspect `embedding/` and `hybrid/` side by side.
2. Focus on whether hybrid introduces visibly more off-topic images in the top-20.
3. Inspect `only_hybrid/` first because those are the images BM25 effectively promoted into view.

Pass condition:

- all 3 queries look at least no worse than the frozen embedding baseline

Fail condition:

- any query clearly shows more off-topic or lower-relevance images in hybrid than in embedding

This is intentionally strict. The burden is on hybrid to avoid degrading the trusted baseline.

## Failure Interpretation

If the experiment fails, the interpretation should be narrow:

- caption generation is not the subject of the failure
- embedding baseline is not the subject of the failure
- the likely problem is either:
  the lexical-view construction, or
  the BM25 signal itself, or
  the limited tail-rerank rule still being too permissive

## Risks

- Even after lexical compression, BM25 may still overvalue generic traffic terms.
- Restricting hybrid to semantic top-20 may make the experiment too conservative to show gains.
- Manual judgment may still disagree across queries because there is no formal ground truth.

These risks are acceptable because the immediate goal is not to maximize recall. The immediate goal is to test whether a constrained hybrid branch can avoid making things worse.

## Review Checklist

- Confirm the baseline semantic report path is frozen and unchanged.
- Confirm the caption JSON path is frozen and unchanged.
- Confirm the lexical-view artifact is derived only from the existing caption text.
- Confirm hybrid top-5 exactly matches embedding top-5 for every query.
- Confirm hybrid never introduces images outside semantic top-20.
- Confirm all outputs are written to new report and comparison paths.
