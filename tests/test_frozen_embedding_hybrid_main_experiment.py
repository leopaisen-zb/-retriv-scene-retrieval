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


def test_write_query_comparison_dirs_uses_item_position_when_rank_missing_or_null(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (image_dir / name).write_bytes(b"fake image")

    embedding_row = {
        "query": "雨天，前车刹车灯亮起",
        "items": [
            {
                "image_id": "a",
                "image_path": str(image_dir / "a.jpg"),
                "rank": None,
                "score": 0.9,
                "caption": "caption for a",
                "source_result_file": "captions.json",
            },
            {
                "image_id": "b",
                "image_path": str(image_dir / "b.jpg"),
                "rank": "bad",
                "score": 0.8,
                "caption": "caption for b",
                "source_result_file": "captions.json",
            },
        ],
    }
    hybrid_row = {
        "query": "雨天，前车刹车灯亮起",
        "items": [
            {
                "image_id": "a",
                "image_path": str(image_dir / "a.jpg"),
                "rank": None,
                "score": 0.9,
                "caption": "caption for a",
                "source_result_file": "captions.json",
            },
            {
                "image_id": "c",
                "image_path": str(image_dir / "c.jpg"),
                "rank": None,
                "score": 0.7,
                "caption": "caption for c",
                "source_result_file": "captions.json",
            },
        ],
    }

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
    assert (query_path / "only_embedding" / "01_b.jpg").exists()
    assert (query_path / "only_hybrid" / "01_c.jpg").exists()
