# Hybrid Filter Rank-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one strict hybrid filtering experiment that keeps the embedding baseline frozen, filters semantic candidates by `embedding top-10 ∩ BM25 top-8`, and writes comparison artifacts that show exactly which images were removed.

**Architecture:** Reuse the current deterministic BM25 lexical view as the lexical signal source. Add one small rank-gate filtering helper that operates on `RetrievalItem` lists, then add one new experiment script that reads the frozen semantic report, runs BM25 on the lexical view, computes the gated intersection, and writes filtered outputs plus comparison folders without touching any existing baseline artifacts.

**Tech Stack:** Python, `rank_bm25`, existing retrieval dataclasses and report helpers, `pytest`, JSON/CSV filesystem outputs.

---

## File Structure

- Create: `src/pipelines/hybrid_rank_gate_filter.py`
  - Pure filtering helper that preserves semantic order and enforces the `top-20 -> top-10 ∩ top-8 -> max-10` contract.
- Create: `tests/test_hybrid_rank_gate_filter.py`
  - Unit tests for the gate logic and validation behavior.
- Create: `src/experiments/frozen_embedding_hybrid_filter_rank_gate_experiment.py`
  - Orchestrates the frozen baseline + lexical BM25 filtering experiment and writes the new report and comparison directories.
- Create: `tests/test_frozen_embedding_hybrid_filter_rank_gate_experiment.py`
  - Locks the manifest schema and query comparison folder structure for the filtering experiment.
- Reuse without modification: `src/utils/bm25_lexical_view.py`
  - Existing deterministic lexical-view builder.
- Reuse without modification: `src/experiments/text_to_image_retrieval_experiment.py`
  - Existing JSON/CSV/image-folder formatting helpers.

Execution note:
- The current sandbox copy is not a Git repository.
- Commit steps below assume implementation happens inside the dedicated git worktree clone recommended by the planning workflow.

### Task 1: Add the rank-gate filtering helper

**Files:**
- Create: `src/pipelines/hybrid_rank_gate_filter.py`
- Create: `tests/test_hybrid_rank_gate_filter.py`

- [ ] **Step 1: Write the failing rank-gate tests**

Create `tests/test_hybrid_rank_gate_filter.py` with:

```python
from __future__ import annotations

import pytest

from pipelines.hybrid_rank_gate_filter import filter_semantic_candidates_by_rank_gates
from pipelines.retrieval_pipeline import RetrievalItem


def _item(image_id: str, score: float) -> RetrievalItem:
    return RetrievalItem(
        image_id=image_id,
        image_path=f"/tmp/{image_id}.jpg",
        score=score,
        caption=f"caption for {image_id}",
        source_result_file="captions.json",
    )


def test_filter_keeps_semantic_order_for_surviving_items() -> None:
    semantic_items = [
        _item("a", 0.91),
        _item("b", 0.82),
        _item("c", 0.73),
        _item("d", 0.64),
    ]
    lexical_items = [
        _item("c", 3.0),
        _item("a", 2.0),
        _item("x", 1.0),
    ]

    filtered = filter_semantic_candidates_by_rank_gates(
        semantic_items=semantic_items,
        lexical_items=lexical_items,
        candidate_top_k=4,
        semantic_top_n=3,
        lexical_top_n=2,
        output_max_k=3,
    )

    assert [item.image_id for item in filtered] == ["a", "c"]


def test_filter_returns_fewer_than_output_max_when_intersection_is_small() -> None:
    semantic_items = [
        _item("a", 0.91),
        _item("b", 0.82),
        _item("c", 0.73),
    ]
    lexical_items = [
        _item("b", 2.0),
        _item("x", 1.0),
    ]

    filtered = filter_semantic_candidates_by_rank_gates(
        semantic_items=semantic_items,
        lexical_items=lexical_items,
        candidate_top_k=3,
        semantic_top_n=3,
        lexical_top_n=2,
        output_max_k=3,
    )

    assert [item.image_id for item in filtered] == ["b"]


def test_filter_never_introduces_lexical_only_items() -> None:
    semantic_items = [
        _item("a", 0.91),
        _item("b", 0.82),
        _item("c", 0.73),
        _item("d", 0.64),
        _item("e", 0.55),
    ]
    lexical_items = [
        _item("x", 5.0),
        _item("b", 4.0),
        _item("y", 3.0),
        _item("c", 2.0),
    ]

    filtered = filter_semantic_candidates_by_rank_gates(
        semantic_items=semantic_items,
        lexical_items=lexical_items,
        candidate_top_k=5,
        semantic_top_n=3,
        lexical_top_n=4,
        output_max_k=3,
    )

    assert [item.image_id for item in filtered] == ["b", "c"]
    assert "x" not in {item.image_id for item in filtered}
    assert "y" not in {item.image_id for item in filtered}


@pytest.mark.parametrize(
    ("candidate_top_k", "semantic_top_n", "lexical_top_n", "output_max_k"),
    [
        (0, 10, 8, 10),
        (20, 0, 8, 10),
        (20, 10, 0, 10),
        (20, 10, 8, 0),
        (5, 6, 3, 3),
    ],
)
def test_filter_validates_gate_bounds(
    candidate_top_k: int,
    semantic_top_n: int,
    lexical_top_n: int,
    output_max_k: int,
) -> None:
    with pytest.raises(ValueError):
        filter_semantic_candidates_by_rank_gates(
            semantic_items=[_item("a", 0.91)],
            lexical_items=[_item("a", 3.0)],
            candidate_top_k=candidate_top_k,
            semantic_top_n=semantic_top_n,
            lexical_top_n=lexical_top_n,
            output_max_k=output_max_k,
        )
```

- [ ] **Step 2: Run the rank-gate tests to verify they fail**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_hybrid_rank_gate_filter.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipelines.hybrid_rank_gate_filter'`.

- [ ] **Step 3: Write the minimal rank-gate implementation**

Create `src/pipelines/hybrid_rank_gate_filter.py` with:

```python
from __future__ import annotations

from pipelines.retrieval_pipeline import RetrievalItem


def filter_semantic_candidates_by_rank_gates(
    semantic_items: list[RetrievalItem],
    lexical_items: list[RetrievalItem],
    candidate_top_k: int = 20,
    semantic_top_n: int = 10,
    lexical_top_n: int = 8,
    output_max_k: int = 10,
) -> list[RetrievalItem]:
    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k 必须为正整数")
    if semantic_top_n <= 0:
        raise ValueError("semantic_top_n 必须为正整数")
    if lexical_top_n <= 0:
        raise ValueError("lexical_top_n 必须为正整数")
    if output_max_k <= 0:
        raise ValueError("output_max_k 必须为正整数")
    if semantic_top_n > candidate_top_k:
        raise ValueError("semantic_top_n 不能大于 candidate_top_k")

    semantic_candidates = semantic_items[:candidate_top_k]
    semantic_gate = semantic_candidates[:semantic_top_n]
    lexical_gate_ids = {
        item.image_id
        for item in lexical_items[:lexical_top_n]
    }

    surviving = [
        item
        for item in semantic_gate
        if item.image_id in lexical_gate_ids
    ]
    return surviving[:output_max_k]
```

- [ ] **Step 4: Run the rank-gate tests to verify they pass**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_hybrid_rank_gate_filter.py
```

Expected: PASS with `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/hybrid_rank_gate_filter.py tests/test_hybrid_rank_gate_filter.py
git commit -m "feat: add hybrid rank-gate filter helper"
```

### Task 2: Add the filtered experiment script and comparison artifacts

**Files:**
- Create: `src/experiments/frozen_embedding_hybrid_filter_rank_gate_experiment.py`
- Create: `tests/test_frozen_embedding_hybrid_filter_rank_gate_experiment.py`

- [ ] **Step 1: Write the failing experiment-script tests**

Create `tests/test_frozen_embedding_hybrid_filter_rank_gate_experiment.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

from experiments.frozen_embedding_hybrid_filter_rank_gate_experiment import (
    build_filter_manifest,
    write_query_filter_comparison_dirs,
)


def _row(query: str, items: list[tuple[int, str, str]]) -> dict[str, object]:
    return {
        "query": query,
        "items": [
            {
                "rank": rank,
                "image_id": image_id,
                "image_path": image_path,
                "score": 1.0 / rank,
                "caption": f"caption for {image_id}",
                "source_result_file": "captions.json",
            }
            for rank, image_id, image_path in items
        ],
    }


def test_build_filter_manifest_tracks_gate_sets_and_removed_ids() -> None:
    embedding_row = _row(
        "低光照下，前车强逆光",
        [
            (1, "a", "/tmp/a.jpg"),
            (2, "b", "/tmp/b.jpg"),
            (3, "c", "/tmp/c.jpg"),
            (4, "d", "/tmp/d.jpg"),
        ],
    )
    bm25_row = _row(
        "低光照下，前车强逆光",
        [
            (1, "b", "/tmp/b.jpg"),
            (2, "x", "/tmp/x.jpg"),
            (3, "a", "/tmp/a.jpg"),
        ],
    )
    filtered_row = _row(
        "低光照下，前车强逆光",
        [
            (1, "a", "/tmp/a.jpg"),
            (2, "b", "/tmp/b.jpg"),
        ],
    )

    manifest = build_filter_manifest(
        query="低光照下，前车强逆光",
        embedding_row=embedding_row,
        bm25_row=bm25_row,
        filtered_row=filtered_row,
        candidate_top_k=4,
        semantic_top_n=3,
        semantic_report_path="semantic.json",
        filtered_report_path="filtered.json",
    )

    assert manifest["embedding_top20_image_ids"] == ["a", "b", "c", "d"]
    assert manifest["embedding_top10_image_ids"] == ["a", "b", "c"]
    assert manifest["bm25_top8_image_ids"] == ["b", "x", "a"]
    assert manifest["final_filtered_image_ids"] == ["a", "b"]
    assert manifest["filtered_out_image_ids"] == ["c"]
    assert manifest["final_result_count"] == 2


def test_write_query_filter_comparison_dirs_creates_expected_subfolders(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        (image_dir / name).write_bytes(b"fake image")

    embedding_row = _row(
        "雨天，前车刹车灯亮起",
        [
            (1, "a", str(image_dir / "a.jpg")),
            (2, "b", str(image_dir / "b.jpg")),
            (3, "c", str(image_dir / "c.jpg")),
            (4, "d", str(image_dir / "d.jpg")),
        ],
    )
    bm25_row = _row(
        "雨天，前车刹车灯亮起",
        [
            (1, "b", str(image_dir / "b.jpg")),
            (2, "d", str(image_dir / "d.jpg")),
        ],
    )
    filtered_row = _row(
        "雨天，前车刹车灯亮起",
        [
            (1, "b", str(image_dir / "b.jpg")),
        ],
    )

    query_dir = write_query_filter_comparison_dirs(
        base_dir=tmp_path / "comparisons",
        embedding_row=embedding_row,
        bm25_row=bm25_row,
        filtered_row=filtered_row,
        candidate_top_k=4,
        semantic_top_n=2,
        semantic_report_path="semantic.json",
        filtered_report_path="filtered.json",
    )

    query_path = Path(query_dir)
    assert (query_path / "embedding_top20" / "01_a.jpg").exists()
    assert (query_path / "embedding_top20" / "04_d.jpg").exists()
    assert (query_path / "embedding_top10" / "01_a.jpg").exists()
    assert (query_path / "embedding_top10" / "02_b.jpg").exists()
    assert (query_path / "bm25_top8_candidates" / "01_b.jpg").exists()
    assert (query_path / "bm25_top8_candidates" / "02_d.jpg").exists()
    assert (query_path / "final_filtered" / "01_b.jpg").exists()
    assert (query_path / "filtered_out_from_embedding_top10" / "01_a.jpg").exists()

    manifest = json.loads((query_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query"] == "雨天，前车刹车灯亮起"
    assert manifest["final_result_count"] == 1
```

- [ ] **Step 2: Run the experiment-script tests to verify they fail**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_frozen_embedding_hybrid_filter_rank_gate_experiment.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.frozen_embedding_hybrid_filter_rank_gate_experiment'`.

- [ ] **Step 3: Write the minimal experiment script**

Create `src/experiments/frozen_embedding_hybrid_filter_rank_gate_experiment.py` with:

```python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.text_to_image_retrieval_experiment import (
    _format_items,
    _organize_images_by_query,
    _sanitize_query_dir_name,
    _to_json_payload,
    _write_recall_summary_csv,
)
from pipelines.hybrid_rank_gate_filter import filter_semantic_candidates_by_rank_gates
from pipelines.retrieval_pipeline import (
    BM25RetrievalBackend,
    CaptionRecord,
    RetrievalItem,
    load_caption_records,
)
from utils.bm25_lexical_view import build_bm25_view_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结 embedding 基线的 rank-gate filtering 实验")
    parser.add_argument(
        "--semantic-report",
        default="output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json",
    )
    parser.add_argument(
        "--caption-result-json",
        default="output/reports/caption_issues_all_qwen35.json",
    )
    parser.add_argument(
        "--bm25-view-json",
        default="output/artifacts/bm25_views/hybrid_filter_rank_gate.json",
    )
    parser.add_argument(
        "--output-json",
        default="output/reports/retrieval_issues_all_hybrid_filter_rank_gate.json",
    )
    parser.add_argument(
        "--output-csv",
        default="output/reports/retrieval_issues_all_hybrid_filter_rank_gate_recall_summary.csv",
    )
    parser.add_argument(
        "--filtered-output-dir",
        default="output/retrieval_output_hybrid_filter_rank_gate",
    )
    parser.add_argument(
        "--comparison-dir",
        default="output/comparisons/hybrid_filter_rank_gate",
    )
    parser.add_argument("--candidate-top-k", type=int, default=20)
    parser.add_argument("--semantic-top-n", type=int, default=10)
    parser.add_argument("--lexical-top-n", type=int, default=8)
    parser.add_argument("--output-max-k", type=int, default=10)
    return parser.parse_args()


def _semantic_items_from_row(row: dict[str, object]) -> list[RetrievalItem]:
    return [
        RetrievalItem(
            image_id=str(item["image_id"]),
            image_path=str(item["image_path"]),
            score=float(item["score"]),
            caption=str(item["caption"]),
            source_result_file=str(item["source_result_file"]),
        )
        for item in row["items"]
    ]


def _lexical_caption_records(
    caption_records: list[CaptionRecord],
) -> tuple[list[CaptionRecord], list[dict[str, object]]]:
    lexical_rows = build_bm25_view_rows(caption_records)
    lexical_records = [
        CaptionRecord(
            image_id=row.image_id,
            image_path=row.image_path,
            caption=row.bm25_text,
            source_result_file="bm25_lexical_view",
        )
        for row in lexical_rows
    ]
    lexical_payload = [
        {
            "image_id": row.image_id,
            "image_path": row.image_path,
            "source_caption": row.source_caption,
            "bm25_text": row.bm25_text,
        }
        for row in lexical_rows
    ]
    return lexical_records, lexical_payload


def build_filter_manifest(
    query: str,
    embedding_row: dict[str, object],
    bm25_row: dict[str, object],
    filtered_row: dict[str, object],
    candidate_top_k: int,
    semantic_top_n: int,
    semantic_report_path: str,
    filtered_report_path: str,
) -> dict[str, object]:
    embedding_top20_items = list(embedding_row["items"])[:candidate_top_k]
    embedding_top10_items = embedding_top20_items[:semantic_top_n]
    bm25_items = list(bm25_row["items"])
    filtered_items = list(filtered_row["items"])

    final_ids = [str(item["image_id"]) for item in filtered_items]
    final_set = set(final_ids)
    embedding_top20_ids = [str(item["image_id"]) for item in embedding_top20_items]
    embedding_top10_ids = [str(item["image_id"]) for item in embedding_top10_items]
    bm25_ids = [str(item["image_id"]) for item in bm25_items]

    return {
        "query": query,
        "semantic_report_path": semantic_report_path,
        "filtered_report_path": filtered_report_path,
        "embedding_top20_image_ids": embedding_top20_ids,
        "embedding_top10_image_ids": embedding_top10_ids,
        "bm25_top8_image_ids": bm25_ids,
        "final_filtered_image_ids": final_ids,
        "filtered_out_image_ids": [
            image_id
            for image_id in embedding_top10_ids
            if image_id not in final_set
        ],
        "final_result_count": len(final_ids),
    }


def _copy_items(items: list[dict[str, object]], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _fallback_rank(value: object, default_rank: int) -> int:
        try:
            rank = int(value)
        except (TypeError, ValueError):
            return default_rank
        return rank if rank > 0 else default_rank

    for rank, item in enumerate(items, start=1):
        src = Path(str(item["image_path"]))
        item_rank = _fallback_rank(item.get("rank", rank), rank)
        shutil.copy2(src, dest_dir / f"{item_rank:02d}_{src.name}")


def write_query_filter_comparison_dirs(
    base_dir: Path,
    embedding_row: dict[str, object],
    bm25_row: dict[str, object],
    filtered_row: dict[str, object],
    candidate_top_k: int,
    semantic_top_n: int,
    semantic_report_path: str,
    filtered_report_path: str,
) -> str:
    query = str(embedding_row["query"])
    query_dir = base_dir / _sanitize_query_dir_name(query)
    if query_dir.exists():
        shutil.rmtree(query_dir)
    query_dir.mkdir(parents=True, exist_ok=True)

    embedding_top20_items = list(embedding_row["items"])[:candidate_top_k]
    embedding_top10_items = embedding_top20_items[:semantic_top_n]
    bm25_items = list(bm25_row["items"])
    filtered_items = list(filtered_row["items"])
    filtered_ids = {str(item["image_id"]) for item in filtered_items}

    _copy_items(embedding_top20_items, query_dir / "embedding_top20")
    _copy_items(embedding_top10_items, query_dir / "embedding_top10")
    _copy_items(bm25_items, query_dir / "bm25_top8_candidates")
    _copy_items(filtered_items, query_dir / "final_filtered")
    _copy_items(
        [
            item
            for item in embedding_top10_items
            if str(item["image_id"]) not in filtered_ids
        ],
        query_dir / "filtered_out_from_embedding_top10",
    )

    manifest = build_filter_manifest(
        query=query,
        embedding_row=embedding_row,
        bm25_row=bm25_row,
        filtered_row=filtered_row,
        candidate_top_k=candidate_top_k,
        semantic_top_n=semantic_top_n,
        semantic_report_path=semantic_report_path,
        filtered_report_path=filtered_report_path,
    )
    (query_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(query_dir.resolve())


def main() -> None:
    args = parse_args()
    semantic_report_path = Path(args.semantic_report)
    semantic_data = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    semantic_rows = semantic_data["query_results"]

    caption_records = load_caption_records(args.caption_result_json)
    lexical_records, lexical_payload = _lexical_caption_records(caption_records)
    lexical_backend = BM25RetrievalBackend(records=lexical_records)

    bm25_view_path = Path(args.bm25_view_json)
    bm25_view_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_view_path.write_text(
        json.dumps({"rows": lexical_payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    filtered_rows: list[dict[str, object]] = []
    bm25_rows: list[dict[str, object]] = []

    for semantic_row in semantic_rows:
        semantic_items = _semantic_items_from_row(semantic_row)
        lexical_items = lexical_backend.search(
            query=str(semantic_row["search_query"]),
            top_k=args.lexical_top_n,
        )
        filtered_items = filter_semantic_candidates_by_rank_gates(
            semantic_items=semantic_items,
            lexical_items=lexical_items,
            candidate_top_k=args.candidate_top_k,
            semantic_top_n=args.semantic_top_n,
            lexical_top_n=args.lexical_top_n,
            output_max_k=args.output_max_k,
        )

        bm25_rows.append(
            {
                "query": semantic_row["query"],
                "items": _format_items(lexical_items[: args.lexical_top_n]),
            }
        )
        filtered_rows.append(
            {
                "query": semantic_row["query"],
                "search_query": semantic_row["search_query"],
                "expanded_english_queries": semantic_row.get("expanded_english_queries", []),
                "items": _format_items(filtered_items),
            }
        )

    payload = _to_json_payload(
        caption_result_json=args.caption_result_json,
        backend="hybrid_embedding_bm25_filter_rank_gate",
        top_k=args.output_max_k,
        query_results=filtered_rows,
    )
    payload["summary"].update(
        {
            "semantic_source_report": str(semantic_report_path),
            "lexical_backend": "bm25",
            "bm25_view_json": str(bm25_view_path),
            "candidate_top_k": args.candidate_top_k,
            "semantic_top_n": args.semantic_top_n,
            "lexical_top_n": args.lexical_top_n,
            "output_max_k": args.output_max_k,
            "candidate_source": "semantic_top_k_only",
            "filter_strategy": "semantic_rank_gate_intersect_bm25_rank_gate",
        }
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_recall_summary_csv(Path(args.output_csv), payload["query_results"])

    created_dirs = _organize_images_by_query(
        Path(args.filtered_output_dir),
        payload["query_results"],
    )
    payload["summary"]["retrieval_output_dir"] = str(Path(args.filtered_output_dir).resolve())
    payload["summary"]["organized_query_dirs"] = created_dirs

    comparison_dirs: list[str] = []
    for embedding_row, bm25_row, filtered_row in zip(
        semantic_rows,
        bm25_rows,
        payload["query_results"],
        strict=True,
    ):
        comparison_dirs.append(
            write_query_filter_comparison_dirs(
                base_dir=Path(args.comparison_dir),
                embedding_row=embedding_row,
                bm25_row=bm25_row,
                filtered_row=filtered_row,
                candidate_top_k=args.candidate_top_k,
                semantic_top_n=args.semantic_top_n,
                semantic_report_path=str(semantic_report_path),
                filtered_report_path=str(out_json),
            )
        )

    payload["summary"]["comparison_dir"] = str(Path(args.comparison_dir).resolve())
    payload["summary"]["comparison_query_dirs"] = comparison_dirs
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the experiment-script tests to verify they pass**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_frozen_embedding_hybrid_filter_rank_gate_experiment.py
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/frozen_embedding_hybrid_filter_rank_gate_experiment.py tests/test_frozen_embedding_hybrid_filter_rank_gate_experiment.py
git commit -m "feat: add frozen hybrid rank-gate filtering experiment"
```

### Task 3: Run the filtering experiment and verify the output contract

**Files:**
- Reuse: `src/utils/bm25_lexical_view.py`
- Reuse: `src/pipelines/hybrid_rank_gate_filter.py`
- Reuse: `src/experiments/frozen_embedding_hybrid_filter_rank_gate_experiment.py`
- Output: `output/artifacts/bm25_views/hybrid_filter_rank_gate.json`
- Output: `output/reports/retrieval_issues_all_hybrid_filter_rank_gate.json`
- Output: `output/reports/retrieval_issues_all_hybrid_filter_rank_gate_recall_summary.csv`
- Output: `output/retrieval_output_hybrid_filter_rank_gate/`
- Output: `output/comparisons/hybrid_filter_rank_gate/`

- [ ] **Step 1: Run the related test bundle**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_bm25_lexical_view.py tests/test_hybrid_rank_gate_filter.py tests/test_frozen_embedding_hybrid_filter_rank_gate_experiment.py
```

Expected: PASS with all tests green.

- [ ] **Step 2: Run the filtering experiment**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src python src/experiments/frozen_embedding_hybrid_filter_rank_gate_experiment.py \
  --semantic-report output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json \
  --caption-result-json output/reports/caption_issues_all_qwen35.json \
  --bm25-view-json output/artifacts/bm25_views/hybrid_filter_rank_gate.json \
  --output-json output/reports/retrieval_issues_all_hybrid_filter_rank_gate.json \
  --output-csv output/reports/retrieval_issues_all_hybrid_filter_rank_gate_recall_summary.csv \
  --filtered-output-dir output/retrieval_output_hybrid_filter_rank_gate \
  --comparison-dir output/comparisons/hybrid_filter_rank_gate \
  --candidate-top-k 20 \
  --semantic-top-n 10 \
  --lexical-top-n 8 \
  --output-max-k 10
```

Expected: exit code `0` and refreshed outputs under the five paths listed above.

- [ ] **Step 3: Verify the filter contract objectively**

Run:

```bash
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path

from experiments.text_to_image_retrieval_experiment import _sanitize_query_dir_name

report = json.loads(
    Path("output/reports/retrieval_issues_all_hybrid_filter_rank_gate.json").read_text(
        encoding="utf-8"
    )
)
comparison_dir = Path("output/comparisons/hybrid_filter_rank_gate")

all_constraints = True
for row in report["query_results"]:
    query = row["query"]
    folder = comparison_dir / _sanitize_query_dir_name(query)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    final_ids = row["recalled_image_ids"]
    embedding_top10 = manifest["embedding_top10_image_ids"]
    bm25_top8 = manifest["bm25_top8_image_ids"]

    if len(final_ids) > 10:
        all_constraints = False
    if any(image_id not in embedding_top10 for image_id in final_ids):
        all_constraints = False
    if any(image_id not in bm25_top8 for image_id in final_ids):
        all_constraints = False

    ordered_top10 = [image_id for image_id in embedding_top10 if image_id in final_ids]
    if ordered_top10 != final_ids:
        all_constraints = False

print("ALL_CONSTRAINTS", all_constraints)
PY
```

Expected:

```text
ALL_CONSTRAINTS True
```

- [ ] **Step 4: Verify the comparison directory layout exists for all queries**

Run:

```bash
find output/comparisons/hybrid_filter_rank_gate -maxdepth 2 -type d | sort
```

Expected: each query directory contains:

```text
embedding_top20
embedding_top10
bm25_top8_candidates
final_filtered
filtered_out_from_embedding_top10
```

- [ ] **Step 5: Skip committing generated runtime outputs**

Do not commit the generated report JSON, CSV, or copied image folders unless the repository already tracks those artifacts and the user explicitly asks for it. In the dedicated git worktree, the only code commit for this plan should come from Tasks 1 and 2.
