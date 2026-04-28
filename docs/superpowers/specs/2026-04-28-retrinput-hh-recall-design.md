# RetrInput HH Recall Experiment Design

Date: 2026-04-28

## Goal

Re-run the retrieval experiment on `RetrInput_hh` and compare vector retrieval against BM25 using recall only. The experiment must use the existing three queries, regenerate captions for the new image set, and report simple top-percentage recall numbers.

## Inputs

- Image root: `RetrInput_hh`
- Query file: `input/processed/scene_queries_specific_en.json`
- Caption output: `output/reports/caption_retrinput_hh_qwen35.json`
- Retrieval/evaluation output: `output/reports/retrinput_hh_embedding_vs_bm25_percent_recall.json`

`RetrInput_hh` contains 99 images grouped into human-labeled folders:

- `低光照下，前车强逆光`: 5 images
- `对向来车`: 56 images
- `雨天`: 26 images
- `雨天，前车刹车灯亮起`: 12 images

The `雨天` folder is a distractor/background class for this experiment. For the rainy brake-light query, strict GT is only `雨天，前车刹车灯亮起`.

## Query To GT Mapping

- `vehicle ahead under strong backlighting in low-light conditions` -> `低光照下，前车强逆光`
- `oncoming vehicle approaching in the opposite lane` -> `对向来车`
- `rainy road scene with the brake lights of the vehicle ahead illuminated` -> `雨天，前车刹车灯亮起`

Search text should use each query file row's `english_query`, matching the existing no-query-expansion experiment behavior.

## Method Comparison

Compare only:

- `qwen3vl_embedding_faiss`
- `bm25`

Do not run or report hybrid/fusion retrieval.

## Metric

Use top-percentage recall, not fixed `top_k` and not absolute score thresholds.

For each query and backend:

1. Rank all 99 images by backend score.
2. Evaluate candidate prefixes at `10%`, `20%`, `30%`, and `50%`.
3. For 99 images, those cutoffs are `10`, `20`, `30`, and `50` images using `ceil(total * percent / 100)`.
4. Compute `recall = recalled_gt_count / gt_count`.

Report per query, backend, and percent cutoff:

- `candidate_count`
- `gt_count`
- `recalled_gt_count`
- `recall`
- `recalled_gt_image_paths`
- `missed_gt_image_paths`

Use full resolved image paths as identity. Do not use `image_id` alone because the same filename can appear in more than one label folder.

## Outputs

The caption JSON should contain all image captions for manual inspection.

The evaluation JSON should contain:

- input paths and percent cutoffs
- query-to-label mapping
- per-query/per-backend recall summaries
- ranked retrieval rows with `is_gt` markers for auditability

The evaluation CSV should contain the compact summary table:

`query,backend,percent,candidate_count,gt_count,recalled_gt_count,recall`

## Validation

Automated tests should cover:

- top-percentage cutoffs use `ceil`
- recall is computed by full image path
- missed and recalled GT sets are reported correctly

Manual/runtime verification should include:

- caption report contains 99 successful rows
- evaluation report includes 3 queries, 2 backends, 4 percent cutoffs
- no hybrid/fusion backend is present in the result

