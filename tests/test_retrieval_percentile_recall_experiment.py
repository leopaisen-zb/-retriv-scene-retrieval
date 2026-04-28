from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path


if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(
        __spec__=ModuleSpec("torch", loader=None),
        dtype=object,
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    )

from experiments.retrieval_percentile_recall_experiment import (
    DEFAULT_QUERY_LABEL_MAP,
    evaluate_percentile_recall,
    percentile_candidate_count,
)
from pipelines.retrieval_pipeline import RetrievalItem


def _item(image_path: Path, rank: int) -> RetrievalItem:
    return RetrievalItem(
        image_id=image_path.stem,
        image_path=str(image_path),
        score=1.0 / rank,
        caption=f"caption {rank}",
        source_result_file="captions.json",
    )


def test_percentile_candidate_count_uses_ceil_and_clamps_to_corpus_size() -> None:
    assert percentile_candidate_count(total_count=99, percent=10) == 10
    assert percentile_candidate_count(total_count=99, percent=20) == 20
    assert percentile_candidate_count(total_count=99, percent=30) == 30
    assert percentile_candidate_count(total_count=99, percent=50) == 50
    assert percentile_candidate_count(total_count=3, percent=1) == 1
    assert percentile_candidate_count(total_count=3, percent=150) == 3


def test_evaluate_percentile_recall_uses_full_paths_not_image_ids(tmp_path: Path) -> None:
    rain_dir = tmp_path / "雨天"
    brake_dir = tmp_path / "雨天，前车刹车灯亮起"
    rain_dir.mkdir()
    brake_dir.mkdir()
    shared_name = "same_name.jpg"
    rain_only = rain_dir / shared_name
    brake_gt = brake_dir / shared_name
    other_gt = brake_dir / "other.jpg"
    distractor = tmp_path / "distractor.jpg"
    for path in (rain_only, brake_gt, other_gt, distractor):
        path.write_bytes(b"fake")

    items = [
        _item(rain_only, 1),
        _item(brake_gt, 2),
        _item(distractor, 3),
        _item(other_gt, 4),
    ]

    rows = evaluate_percentile_recall(
        items=items,
        gt_paths={str(brake_gt.resolve()), str(other_gt.resolve())},
        percentiles=[50, 100],
        total_count=4,
    )

    assert rows[0]["candidate_count"] == 2
    assert rows[0]["recalled_gt_count"] == 1
    assert rows[0]["recall"] == 0.5
    assert rows[0]["recalled_gt_image_paths"] == [str(brake_gt.resolve())]
    assert rows[0]["missed_gt_image_paths"] == [str(other_gt.resolve())]
    assert rows[1]["candidate_count"] == 4
    assert rows[1]["recalled_gt_count"] == 2
    assert rows[1]["recall"] == 1.0


def test_default_query_label_map_uses_strict_rainy_brake_light_gt() -> None:
    assert (
        DEFAULT_QUERY_LABEL_MAP["vehicle ahead under strong backlighting in low-light conditions"]
        == "低光照下，前车强逆光"
    )
    assert (
        DEFAULT_QUERY_LABEL_MAP["oncoming vehicle approaching in the opposite lane"]
        == "对向来车"
    )
    assert (
        DEFAULT_QUERY_LABEL_MAP[
            "rainy road scene with the brake lights of the vehicle ahead illuminated"
        ]
        == "雨天，前车刹车灯亮起"
    )
