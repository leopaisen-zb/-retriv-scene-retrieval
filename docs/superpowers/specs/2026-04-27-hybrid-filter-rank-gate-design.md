# Hybrid Filter Rank-Gate Design

Date: 2026-04-27
Workspace: `/home/leo494/projects/Retriv`

## Goal

Run one new hybrid retrieval experiment whose only question is:

Can a strict hybrid filter produce fewer but more accurate results than the frozen embedding baseline under manual image review?

This experiment does not optimize recall. It explicitly allows the final result count to drop below `top_k` if that improves precision.

## Non-Negotiable Constraints

The following inputs remain frozen and must not be modified:

- embedding retrieval configuration
- existing embedding result files
- caption result JSON
- caption prompt and prompt configuration

Specifically:

- Frozen embedding report:
  `output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json`
- Frozen caption corpus:
  `output/reports/caption_issues_all_qwen35.json`

The experiment may create new files and new folders, but it must not overwrite or mutate the baseline chain.

## Baseline Definition

The semantic baseline remains the existing embedding retrieval run:

- backend: `qwen3vl_embedding_faiss`
- embedding model: `Qwen3-VL-Embedding-2B`
- retrieval corpus: caption text loaded from `caption_issues_all_qwen35.json`
- baseline result size: `top_k=20`

For this experiment, the baseline report is treated as immutable input. The new hybrid branch may read it and derive filtered outputs from it, but it may not reconfigure, replace, or regenerate it.

## Experiment Scope

In scope:

- reuse the current deterministic BM25 lexical-view approach
- run one filtering-style hybrid experiment
- allow the final result count to be less than `top_k`
- write all outputs to new files and folders
- produce comparison folders that make filtering decisions easy to inspect by eye

Out of scope:

- changing caption generation
- changing caption prompts
- changing the embedding model
- changing baseline embedding CLI defaults
- rerunning the baseline to obtain a different semantic ranking
- mixing filtering and reranking into the same experiment

## Architecture

The new branch should have this shape:

```text
frozen caption JSON -> deterministic BM25 lexical view -> BM25 rank list
frozen embedding report -------------------------------> embedding rank list
embedding rank gate + BM25 rank gate -----------------> filtered final set
filtered final set -----------------------------------> new report + comparison folders
```

Key principle:

This experiment is not a reranker. It is a gate-based filter. BM25 is used to remove weak semantic matches, not to reshuffle the ordering of surviving items.

## BM25 Lexical View

The BM25 side continues to use a derived lexical view built from the frozen caption text.

Requirements:

- deterministic transformation only
- no new model inference
- no rewriting of the original caption file
- stored as a separate artifact for inspection

The lexical view should continue to remove obvious retrieval noise such as:

- timestamps
- camera channel markers
- plate-like tokens
- heading-like caption labels

The lexical view should preserve short scene-bearing tokens that help filtering:

- low-light and night cues
- glare and strong-light cues
- lead-vehicle cues
- wet-road and reflection cues
- brake-light cues

The lexical-view artifact should contain:

- `image_id`
- `image_path`
- `source_caption`
- `bm25_text`

## Filtering Strategy

This experiment uses one strict filtering rule only.

Candidate pool:

- take the frozen embedding `top-20` as the only candidate set

Rank gates:

- semantic gate: keep only items with `embedding rank <= 10`
- lexical gate: keep only items with `BM25 rank <= 8`

Final set:

- `final = (embedding top-10) ∩ (bm25 top-8)`

Output policy:

- keep the final output in original embedding rank order
- do not rerank surviving items
- maximum output size is `10`
- minimum output size is `0`
- do not backfill filtered results with weaker candidates

This means:

- if the intersection has `7` items, return `7`
- if the intersection has `2` items, return `2`
- if the intersection is empty, return an empty result

## Rationale

This design matches the new objective:

- precision matters more than recall
- fewer results are acceptable
- BM25 should act as a hard relevance filter, not a soft reorder signal

Using rank gates instead of score thresholds keeps the rule simple and query-stable:

- embedding scores are not guaranteed to calibrate well across queries
- BM25 raw scores are even harder to compare across different query texts
- rank-based gates are easy to inspect and explain during manual review

## Data Flow

1. Read the frozen embedding report.
2. Read the frozen caption JSON.
3. Build or refresh a BM25 lexical-view artifact dedicated to this experiment.
4. Run BM25 search for each query on that lexical view.
5. For each query, identify:
   - embedding top-20
   - embedding top-10
   - BM25 top-8
6. Compute the intersection of embedding top-10 and BM25 top-8.
7. Preserve embedding order for the surviving items.
8. Write the filtered report and comparison folders.

## Output Contract

This experiment must write only new outputs.

Recommended paths:

- BM25 lexical view:
  `output/artifacts/bm25_views/hybrid_filter_rank_gate.json`
- Filtered report:
  `output/reports/retrieval_issues_all_hybrid_filter_rank_gate.json`
- Filtered summary CSV:
  `output/reports/retrieval_issues_all_hybrid_filter_rank_gate_recall_summary.csv`
- Filtered image folders:
  `output/retrieval_output_hybrid_filter_rank_gate/`
- Comparison artifacts:
  `output/comparisons/hybrid_filter_rank_gate/`

Per query, the comparison folder should include:

- `embedding_top20/`
- `embedding_top10/`
- `bm25_top8_candidates/`
- `final_filtered/`
- `filtered_out_from_embedding_top10/`
- `manifest.json`

`manifest.json` should record, at minimum:

- query text
- frozen embedding report path
- filtered hybrid report path
- embedding top-20 image ids
- embedding top-10 image ids
- BM25 top-8 image ids
- final filtered image ids
- filtered-out image ids
- final result count

## Manual Review Protocol

Success is judged by image inspection, not by automatic metrics.

Review process for each query:

1. Inspect `embedding_top10/` as the semantic baseline head.
2. Inspect `final_filtered/` as the strict filtered result.
3. Inspect `filtered_out_from_embedding_top10/` to judge whether the removed images were correctly discarded.
4. Inspect `bm25_top8_candidates/` only as supporting evidence for why the filter kept or removed images.

Pass condition:

- the filtered result set looks clearly tighter or more accurate than the baseline head, even if it is smaller

Fail condition:

- the filter removes obviously relevant images
- or the filtered set still contains too many off-topic images to justify the loss in recall

## Failure Interpretation

If this experiment fails, the interpretation should stay narrow:

- caption generation is not the subject of the failure
- embedding baseline is not the subject of the failure
- the likely problem is one of:
  - BM25 lexical view still too noisy
  - the `top-10 ∩ top-8` gate is too strict
  - BM25 rank is not a reliable filter for these scene descriptions

## Risks

- The gate may be too strict for some queries and return very few images.
- BM25 may still overvalue generic night-driving or strong-light scenes that are not truly on target.
- Because the final order stays semantic-only, the experiment cannot improve within-set ranking; it can only improve set quality.
