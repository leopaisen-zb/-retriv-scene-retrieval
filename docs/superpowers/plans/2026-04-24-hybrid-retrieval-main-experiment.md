# Hybrid Retrieval Main Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one conservative hybrid retrieval experiment that keeps the embedding baseline frozen, derives a BM25-only lexical view from the existing caption corpus, and writes side-by-side comparison artifacts for manual review.

**Architecture:** Keep the current semantic baseline report as an immutable input. Add one deterministic lexical-view builder for BM25, one constrained tail-rerank function that freezes semantic top-5 and only reorders semantic ranks 6-20, and one new experiment script that writes a hybrid report plus comparison folders without touching any existing embedding or caption artifact.

**Tech Stack:** Python, `rank_bm25`, existing retrieval dataclasses and report helpers, `pytest`, JSON/CSV filesystem outputs.

---

## File Structure

- Create: `src/utils/bm25_lexical_view.py`
  - Deterministically convert long caption text into a short BM25-friendly lexical text.
- Create: `src/pipelines/hybrid_tail_rerank.py`
  - Freeze semantic top-5 and only rerank semantic ranks 6-20 using BM25 support.
- Create: `src/experiments/frozen_embedding_hybrid_main_experiment.py`
  - Orchestrate the experiment from frozen semantic report + frozen caption JSON to new hybrid outputs.
- Create: `tests/test_bm25_lexical_view.py`
  - Lock lexical-view cleaning behavior.
- Create: `tests/test_hybrid_tail_rerank.py`
  - Lock the conservative rerank contract.
- Create: `tests/test_frozen_embedding_hybrid_main_experiment.py`
  - Lock comparison-manifest and comparison-folder behavior.
- Modify: `docs/dev/experiment_run_record.md`
  - Append the hybrid main experiment command, output paths, and manual review notes.

Execution note:
- The current sandbox copy is not a Git repository.
- Commit steps below assume implementation happens inside the dedicated git worktree clone recommended by the planning workflow.

### Task 1: Build the BM25 lexical-view generator

**Files:**
- Create: `src/utils/bm25_lexical_view.py`
- Test: `tests/test_bm25_lexical_view.py`

- [ ] **Step 1: Write the failing lexical-view tests**

Create `tests/test_bm25_lexical_view.py` with:

```python
from __future__ import annotations

from pipelines.retrieval_pipeline import CaptionRecord
from utils.bm25_lexical_view import build_bm25_text, build_bm25_view_rows


def test_build_bm25_text_removes_timestamp_channel_and_section_labels() -> None:
    caption = """
    Timestamp 2023-04-25 13:35:34 CH4
    **Foreground**
    Wet road with the brake lights of the lead vehicle illuminated.
    Overall Composition: city traffic scene.
    """

    text = build_bm25_text(caption)

    assert "2023" not in text
    assert "ch4" not in text
    assert "foreground" not in text
    assert "overall" not in text
    assert "wet" in text
    assert "brake" in text


def test_build_bm25_text_keeps_scene_terms_and_deduplicates_tokens() -> None:
    caption = """
    Rainy road scene with wet reflective pavement.
    A lead vehicle ahead shows bright brake lights.
    The wet road reflects the brake lights.
    """

    text = build_bm25_text(caption)
    tokens = text.split()

    assert "rainy" in tokens
    assert "wet" in tokens
    assert "brake" in tokens
    assert tokens.count("wet") == 1


def test_build_bm25_view_rows_preserves_record_identity() -> None:
    records = [
        CaptionRecord(
            image_id="img-1",
            image_path="/tmp/img-1.jpg",
            caption="Night scene with strong glare from the vehicle ahead.",
            source_result_file="captions.json",
        )
    ]

    rows = build_bm25_view_rows(records)

    assert len(rows) == 1
    assert rows[0].image_id == "img-1"
    assert rows[0].image_path == "/tmp/img-1.jpg"
    assert rows[0].source_caption == records[0].caption
    assert "glare" in rows[0].bm25_text.split()
```

- [ ] **Step 2: Run the lexical-view tests to verify they fail**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_bm25_lexical_view.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'utils.bm25_lexical_view'`.

- [ ] **Step 3: Write the minimal lexical-view implementation**

Create `src/utils/bm25_lexical_view.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
import re

from pipelines.retrieval_pipeline import CaptionRecord

_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b")
_CHANNEL_RE = re.compile(r"\bch\d+\b", re.IGNORECASE)
_PLATE_RE = re.compile(r"\b[\u4e00-\u9fff]?[A-Z]{1,3}[A-Z0-9]{4,6}\b")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_SECTION_LABELS = (
    "foreground",
    "background",
    "mid-ground",
    "overall composition",
    "immediate surroundings",
    "lighting",
    "atmosphere",
    "specific details",
)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "with",
    "and",
    "from",
    "into",
    "over",
    "under",
    "scene",
    "image",
    "vehicle",
}
_SCENE_TERMS = (
    "rain",
    "rainy",
    "wet",
    "reflection",
    "reflective",
    "brake",
    "taillight",
    "tail",
    "light",
    "lights",
    "low",
    "night",
    "glare",
    "backlight",
    "backlighting",
    "oncoming",
    "opposite",
    "lane",
    "ahead",
    "lead",
    "headlight",
    "headlights",
    "visibility",
)


@dataclass(frozen=True)
class BM25LexicalViewRow:
    image_id: str
    image_path: str
    source_caption: str
    bm25_text: str


def _normalize_caption(source_caption: str) -> str:
    text = source_caption.lower()
    text = _TIMESTAMP_RE.sub(" ", text)
    text = _CHANNEL_RE.sub(" ", text)
    text = _PLATE_RE.sub(" ", text)
    text = text.replace("*", " ")
    return re.sub(r"\s+", " ", text).strip()


def build_bm25_text(source_caption: str, max_tokens: int = 48) -> str:
    cleaned = _normalize_caption(source_caption)
    kept_phrases: list[str] = []

    for raw_phrase in re.split(r"[.\n]+", cleaned):
        phrase = raw_phrase.strip(" :-")
        if not phrase:
            continue
        if any(label in phrase for label in _SECTION_LABELS):
            continue
        if any(term in phrase for term in _SCENE_TERMS):
            kept_phrases.append(phrase)

    source = " ".join(kept_phrases) if kept_phrases else cleaned
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(source):
        if token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
        if len(tokens) >= max_tokens:
            break
    return " ".join(tokens)


def build_bm25_view_rows(records: list[CaptionRecord]) -> list[BM25LexicalViewRow]:
    return [
        BM25LexicalViewRow(
            image_id=record.image_id,
            image_path=record.image_path,
            source_caption=record.caption,
            bm25_text=build_bm25_text(record.caption),
        )
        for record in records
    ]
```

- [ ] **Step 4: Run the lexical-view tests to verify they pass**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_bm25_lexical_view.py
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/utils/bm25_lexical_view.py tests/test_bm25_lexical_view.py
git commit -m "feat: add deterministic bm25 lexical view builder"
```

### Task 2: Lock the conservative tail-rerank contract

**Files:**
- Create: `src/pipelines/hybrid_tail_rerank.py`
- Test: `tests/test_hybrid_tail_rerank.py`

- [ ] **Step 1: Write the failing tail-rerank tests**

Create `tests/test_hybrid_tail_rerank.py` with:

```python
from __future__ import annotations

from pipelines.hybrid_tail_rerank import rerank_semantic_tail_with_lexical_support
from pipelines.retrieval_pipeline import RetrievalItem


def _item(image_id: str) -> RetrievalItem:
    return RetrievalItem(
        image_id=image_id,
        image_path=f"/tmp/{image_id}.jpg",
        score=1.0,
        caption=f"caption for {image_id}",
        source_result_file="captions.json",
    )


def test_rerank_keeps_semantic_top5_exact() -> None:
    semantic = [_item(str(i)) for i in range(1, 11)]
    lexical = [_item("8"), _item("7"), _item("6")]

    merged = rerank_semantic_tail_with_lexical_support(
        semantic_items=semantic,
        lexical_items=lexical,
        freeze_top_n=5,
        final_top_k=10,
        lexical_top_n=3,
    )

    assert [item.image_id for item in merged[:5]] == ["1", "2", "3", "4", "5"]


def test_rerank_never_introduces_lexical_only_candidates() -> None:
    semantic = [_item(str(i)) for i in range(1, 11)]
    lexical = [_item("external"), _item("8"), _item("7")]

    merged = rerank_semantic_tail_with_lexical_support(
        semantic_items=semantic,
        lexical_items=lexical,
        freeze_top_n=5,
        final_top_k=10,
        lexical_top_n=3,
    )

    assert "external" not in [item.image_id for item in merged]


def test_rerank_promotes_only_supported_tail_items() -> None:
    semantic = [_item(str(i)) for i in range(1, 11)]
    lexical = [_item("9"), _item("7"), _item("6")]

    merged = rerank_semantic_tail_with_lexical_support(
        semantic_items=semantic,
        lexical_items=lexical,
        freeze_top_n=5,
        final_top_k=10,
        lexical_top_n=3,
    )

    assert [item.image_id for item in merged] == ["1", "2", "3", "4", "5", "9", "7", "6", "8", "10"]
```

- [ ] **Step 2: Run the tail-rerank tests to verify they fail**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_hybrid_tail_rerank.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pipelines.hybrid_tail_rerank'`.

- [ ] **Step 3: Write the minimal tail-rerank implementation**

Create `src/pipelines/hybrid_tail_rerank.py` with:

```python
from __future__ import annotations

from pipelines.retrieval_pipeline import RetrievalItem


def rerank_semantic_tail_with_lexical_support(
    semantic_items: list[RetrievalItem],
    lexical_items: list[RetrievalItem],
    freeze_top_n: int = 5,
    final_top_k: int = 20,
    lexical_top_n: int = 10,
) -> list[RetrievalItem]:
    if freeze_top_n < 0:
        raise ValueError("freeze_top_n 不能为负数")
    if final_top_k <= 0:
        raise ValueError("final_top_k 必须为正整数")
    if lexical_top_n <= 0:
        raise ValueError("lexical_top_n 必须为正整数")

    semantic_slice = semantic_items[:final_top_k]
    frozen = semantic_slice[:freeze_top_n]
    tail = semantic_slice[freeze_top_n:]
    semantic_rank = {item.image_id: idx for idx, item in enumerate(tail)}
    lexical_rank = {
        item.image_id: rank
        for rank, item in enumerate(lexical_items[:lexical_top_n], start=1)
    }

    reranked_tail = sorted(
        tail,
        key=lambda item: (
            1 if item.image_id not in lexical_rank else 0,
            lexical_rank.get(item.image_id, 10**9),
            semantic_rank[item.image_id],
        ),
    )
    return frozen + reranked_tail
```

- [ ] **Step 4: Run the tail-rerank tests to verify they pass**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_hybrid_tail_rerank.py
```

Expected: PASS with `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/pipelines/hybrid_tail_rerank.py tests/test_hybrid_tail_rerank.py
git commit -m "feat: add conservative hybrid tail rerank"
```

### Task 3: Add the frozen-baseline hybrid experiment script and comparison outputs

**Files:**
- Create: `src/experiments/frozen_embedding_hybrid_main_experiment.py`
- Test: `tests/test_frozen_embedding_hybrid_main_experiment.py`

- [ ] **Step 1: Write the failing experiment-helper tests**

Create `tests/test_frozen_embedding_hybrid_main_experiment.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

from experiments.frozen_embedding_hybrid_main_experiment import (
    build_comparison_manifest,
    write_query_comparison_dirs,
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


def test_build_comparison_manifest_lists_overlap_and_uniques() -> None:
    embedding_row = _row(
        "对向来车",
        [(1, "a", "/tmp/a.jpg"), (2, "b", "/tmp/b.jpg"), (3, "c", "/tmp/c.jpg")],
    )
    hybrid_row = _row(
        "对向来车",
        [(1, "a", "/tmp/a.jpg"), (2, "c", "/tmp/c.jpg"), (3, "d", "/tmp/d.jpg")],
    )

    manifest = build_comparison_manifest(
        query="对向来车",
        embedding_row=embedding_row,
        hybrid_row=hybrid_row,
        semantic_report_path="semantic.json",
        hybrid_report_path="hybrid.json",
    )

    assert manifest["intersection_image_ids"] == ["a", "c"]
    assert manifest["only_embedding_image_ids"] == ["b"]
    assert manifest["only_hybrid_image_ids"] == ["d"]


def test_write_query_comparison_dirs_creates_subfolders_and_manifest(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (image_dir / name).write_bytes(b"fake image")

    embedding_row = _row(
        "雨天，前车刹车灯亮起",
        [
            (1, "a", str(image_dir / "a.jpg")),
            (2, "b", str(image_dir / "b.jpg")),
        ],
    )
    hybrid_row = _row(
        "雨天，前车刹车灯亮起",
        [
            (1, "a", str(image_dir / "a.jpg")),
            (2, "c", str(image_dir / "c.jpg")),
        ],
    )

    query_dir = write_query_comparison_dirs(
        base_dir=tmp_path / "comparisons",
        embedding_row=embedding_row,
        hybrid_row=hybrid_row,
        semantic_report_path="semantic.json",
        hybrid_report_path="hybrid.json",
    )

    query_path = Path(query_dir)
    assert (query_path / "embedding" / "01_a.jpg").exists()
    assert (query_path / "embedding" / "02_b.jpg").exists()
    assert (query_path / "hybrid" / "01_a.jpg").exists()
    assert (query_path / "hybrid" / "02_c.jpg").exists()
    assert (query_path / "only_embedding" / "02_b.jpg").exists()
    assert (query_path / "only_hybrid" / "02_c.jpg").exists()

    manifest = json.loads((query_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query"] == "雨天，前车刹车灯亮起"
```

- [ ] **Step 2: Run the experiment-helper tests to verify they fail**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_frozen_embedding_hybrid_main_experiment.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.frozen_embedding_hybrid_main_experiment'`.

- [ ] **Step 3: Write the experiment script**

Create `src/experiments/frozen_embedding_hybrid_main_experiment.py` with:

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
from pipelines.hybrid_tail_rerank import rerank_semantic_tail_with_lexical_support
from pipelines.retrieval_pipeline import BM25RetrievalBackend, CaptionRecord, RetrievalItem, load_caption_records
from utils.bm25_lexical_view import build_bm25_view_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结 embedding 基线的 hybrid 主实验")
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
        default="output/artifacts/bm25_views/hybrid_main_experiment.json",
    )
    parser.add_argument(
        "--output-json",
        default="output/reports/retrieval_issues_all_hybrid_main_experiment.json",
    )
    parser.add_argument(
        "--output-csv",
        default="output/reports/retrieval_issues_all_hybrid_main_experiment_recall_summary.csv",
    )
    parser.add_argument(
        "--hybrid-output-dir",
        default="output/retrieval_output_hybrid_main_experiment",
    )
    parser.add_argument(
        "--comparison-dir",
        default="output/comparisons/hybrid_main_experiment",
    )
    parser.add_argument("--freeze-top-n", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--lexical-top-n", type=int, default=10)
    return parser.parse_args()


def _semantic_items_from_row(row: dict[str, object]) -> list[RetrievalItem]:
    return [
        RetrievalItem(
            image_id=item["image_id"],
            image_path=item["image_path"],
            score=float(item["score"]),
            caption=item["caption"],
            source_result_file=item["source_result_file"],
        )
        for item in row["items"]
    ]


def _lexical_caption_records(caption_records: list[CaptionRecord]) -> tuple[list[CaptionRecord], list[dict[str, object]]]:
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


def build_comparison_manifest(
    query: str,
    embedding_row: dict[str, object],
    hybrid_row: dict[str, object],
    semantic_report_path: str,
    hybrid_report_path: str,
) -> dict[str, object]:
    embedding_ids = [item["image_id"] for item in embedding_row["items"]]
    hybrid_ids = [item["image_id"] for item in hybrid_row["items"]]
    hybrid_set = set(hybrid_ids)
    embedding_set = set(embedding_ids)
    return {
        "query": query,
        "semantic_report_path": semantic_report_path,
        "hybrid_report_path": hybrid_report_path,
        "embedding_image_ids": embedding_ids,
        "hybrid_image_ids": hybrid_ids,
        "intersection_image_ids": [image_id for image_id in hybrid_ids if image_id in embedding_set],
        "only_embedding_image_ids": [image_id for image_id in embedding_ids if image_id not in hybrid_set],
        "only_hybrid_image_ids": [image_id for image_id in hybrid_ids if image_id not in embedding_set],
    }


def _copy_items(items: list[dict[str, object]], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        src = Path(str(item["image_path"]))
        rank = int(item["rank"])
        dest_path = dest_dir / f"{rank:02d}_{src.name}"
        shutil.copy2(src, dest_path)


def write_query_comparison_dirs(
    base_dir: Path,
    embedding_row: dict[str, object],
    hybrid_row: dict[str, object],
    semantic_report_path: str,
    hybrid_report_path: str,
) -> str:
    query = str(embedding_row["query"])
    query_dir = base_dir / _sanitize_query_dir_name(query)
    if query_dir.exists():
        shutil.rmtree(query_dir)
    query_dir.mkdir(parents=True, exist_ok=True)

    embedding_items = list(embedding_row["items"])
    hybrid_items = list(hybrid_row["items"])
    hybrid_ids = {item["image_id"] for item in hybrid_items}
    embedding_ids = {item["image_id"] for item in embedding_items}

    _copy_items(embedding_items, query_dir / "embedding")
    _copy_items(hybrid_items, query_dir / "hybrid")
    _copy_items([item for item in embedding_items if item["image_id"] not in hybrid_ids], query_dir / "only_embedding")
    _copy_items([item for item in hybrid_items if item["image_id"] not in embedding_ids], query_dir / "only_hybrid")

    manifest = build_comparison_manifest(
        query=query,
        embedding_row=embedding_row,
        hybrid_row=hybrid_row,
        semantic_report_path=semantic_report_path,
        hybrid_report_path=hybrid_report_path,
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
    bm25_view_path.write_text(json.dumps({"rows": lexical_payload}, ensure_ascii=False, indent=2), encoding="utf-8")

    hybrid_rows: list[dict[str, object]] = []
    for semantic_row in semantic_rows:
        semantic_items = _semantic_items_from_row(semantic_row)
        lexical_items = lexical_backend.search(
            query=str(semantic_row["search_query"]),
            top_k=max(args.top_k, args.lexical_top_n),
        )
        merged_items = rerank_semantic_tail_with_lexical_support(
            semantic_items=semantic_items,
            lexical_items=lexical_items,
            freeze_top_n=args.freeze_top_n,
            final_top_k=args.top_k,
            lexical_top_n=args.lexical_top_n,
        )
        hybrid_rows.append(
            {
                "query": semantic_row["query"],
                "search_query": semantic_row["search_query"],
                "expanded_english_queries": semantic_row.get("expanded_english_queries", []),
                "items": _format_items(merged_items),
            }
        )

    payload = _to_json_payload(
        caption_result_json=args.caption_result_json,
        backend="hybrid_embedding_bm25_main_experiment",
        top_k=args.top_k,
        query_results=hybrid_rows,
    )
    payload["summary"].update(
        {
            "semantic_source_report": str(semantic_report_path),
            "lexical_backend": "bm25",
            "bm25_view_json": str(bm25_view_path),
            "freeze_top_n": args.freeze_top_n,
            "lexical_top_n": args.lexical_top_n,
            "candidate_source": "semantic_top_k_only",
        }
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_recall_summary_csv(Path(args.output_csv), payload["query_results"])
    created_dirs = _organize_images_by_query(Path(args.hybrid_output_dir), payload["query_results"])
    payload["summary"]["retrieval_output_dir"] = str(Path(args.hybrid_output_dir).resolve())
    payload["summary"]["organized_query_dirs"] = created_dirs

    comparison_dirs = []
    for embedding_row, hybrid_row in zip(semantic_rows, payload["query_results"], strict=True):
        comparison_dirs.append(
            write_query_comparison_dirs(
                base_dir=Path(args.comparison_dir),
                embedding_row=embedding_row,
                hybrid_row=hybrid_row,
                semantic_report_path=str(semantic_report_path),
                hybrid_report_path=str(out_json),
            )
        )
    payload["summary"]["comparison_dir"] = str(Path(args.comparison_dir).resolve())
    payload["summary"]["comparison_query_dirs"] = comparison_dirs
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the experiment-helper tests to verify they pass**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_frozen_embedding_hybrid_main_experiment.py
```

Expected: PASS with `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/frozen_embedding_hybrid_main_experiment.py tests/test_frozen_embedding_hybrid_main_experiment.py
git commit -m "feat: add frozen-baseline hybrid experiment script"
```

### Task 4: Run the main experiment and record the outputs

**Files:**
- Modify: `docs/dev/experiment_run_record.md`
- Runtime output: `output/artifacts/bm25_views/hybrid_main_experiment.json`
- Runtime output: `output/reports/retrieval_issues_all_hybrid_main_experiment.json`
- Runtime output: `output/reports/retrieval_issues_all_hybrid_main_experiment_recall_summary.csv`
- Runtime output: `output/retrieval_output_hybrid_main_experiment/`
- Runtime output: `output/comparisons/hybrid_main_experiment/`

- [ ] **Step 1: Run the unit-test bundle before the experiment**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src pytest -q tests/test_bm25_lexical_view.py tests/test_hybrid_tail_rerank.py tests/test_frozen_embedding_hybrid_main_experiment.py
```

Expected: PASS with all tests green.

- [ ] **Step 2: Execute the hybrid main experiment**

Run:

```bash
source /home/leo494/miniforge3/etc/profile.d/conda.sh && conda activate dc25 && PYTHONPATH=src python src/experiments/frozen_embedding_hybrid_main_experiment.py \
  --semantic-report output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json \
  --caption-result-json output/reports/caption_issues_all_qwen35.json \
  --bm25-view-json output/artifacts/bm25_views/hybrid_main_experiment.json \
  --output-json output/reports/retrieval_issues_all_hybrid_main_experiment.json \
  --output-csv output/reports/retrieval_issues_all_hybrid_main_experiment_recall_summary.csv \
  --hybrid-output-dir output/retrieval_output_hybrid_main_experiment \
  --comparison-dir output/comparisons/hybrid_main_experiment \
  --freeze-top-n 5 \
  --top-k 20 \
  --lexical-top-n 10
```

Expected: exit code `0`, with all six output paths created.

- [ ] **Step 3: Verify the frozen-baseline contract**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

embedding = json.loads(Path("output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json").read_text())
hybrid = json.loads(Path("output/reports/retrieval_issues_all_hybrid_main_experiment.json").read_text())
for emb_row, hyb_row in zip(embedding["query_results"], hybrid["query_results"], strict=True):
    emb_top5 = [item["image_id"] for item in emb_row["items"][:5]]
    hyb_top5 = [item["image_id"] for item in hyb_row["items"][:5]]
    assert emb_top5 == hyb_top5, (emb_row["query"], emb_top5, hyb_top5)
print("top5 frozen contract verified")
PY
```

Expected: prints `top5 frozen contract verified`.

- [ ] **Step 4: Append the hybrid experiment record**

Append this section to `docs/dev/experiment_run_record.md`:

```md
## 2026-04-24 Frozen-Baseline Hybrid Main Experiment

### 实验背景

- 目标：在不修改 embedding 基线、不修改 caption、不修改 caption prompt 的前提下，验证 conservative hybrid 是否至少不比 embedding 差。
- 语义基线：`output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json`
- caption 语料：`output/reports/caption_issues_all_qwen35.json`
- lexical 语料：`output/artifacts/bm25_views/hybrid_main_experiment.json`

### 实验配置

- rerank 策略：冻结 top-5，仅允许重排 semantic top-20 内的 rank 6-20
- lexical backend：`bm25`
- freeze_top_n：`5`
- top_k：`20`
- lexical_top_n：`10`
- hybrid 输出：`output/reports/retrieval_issues_all_hybrid_main_experiment.json`
- comparison 目录：`output/comparisons/hybrid_main_experiment/`

### 人工复核记录

- 低光照下，前车强逆光：写入实际人工复核结论，例如 `至少不差`、`更差` 或 `不确定`
- 对向来车：写入实际人工复核结论，例如 `至少不差`、`更差` 或 `不确定`
- 雨天，前车刹车灯亮起：写入实际人工复核结论，例如 `至少不差`、`更差` 或 `不确定`

### 结论

- 总体判断：写入实际总评结论，例如 `通过`、`失败` 或 `需要下一轮`
```

- [ ] **Step 5: Commit**

```bash
git add docs/dev/experiment_run_record.md \
  src/utils/bm25_lexical_view.py \
  src/pipelines/hybrid_tail_rerank.py \
  src/experiments/frozen_embedding_hybrid_main_experiment.py \
  tests/test_bm25_lexical_view.py \
  tests/test_hybrid_tail_rerank.py \
  tests/test_frozen_embedding_hybrid_main_experiment.py
git commit -m "feat: add frozen-baseline hybrid main experiment"
```
