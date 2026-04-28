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

from experiments.strict_dedup_recall_experiment import (
    build_deduplicated_corpus,
    evaluate_fixed_topk_hash_recall,
)
from pipelines.retrieval_pipeline import CaptionRecord, RetrievalItem


def _record(path: Path, caption: str) -> CaptionRecord:
    return CaptionRecord(
        image_id=path.stem,
        image_path=str(path),
        caption=caption,
        source_result_file="captions.json",
    )


def _item(path: Path, rank: int) -> RetrievalItem:
    return RetrievalItem(
        image_id=path.stem,
        image_path=str(path),
        score=1.0 / rank,
        caption=f"caption {rank}",
        source_result_file="captions.json",
    )


def test_build_deduplicated_corpus_uses_content_hash_and_preserves_full_label_membership(
    tmp_path: Path,
) -> None:
    label_a = tmp_path / "对向来车"
    label_b = tmp_path / "低光照下，前车强逆光"
    label_a.mkdir()
    label_b.mkdir()
    duplicate_a = label_a / "same-a.jpg"
    duplicate_b = label_b / "same-b.jpg"
    unique = label_a / "unique.jpg"
    duplicate_a.write_bytes(b"same-image-content")
    duplicate_b.write_bytes(b"same-image-content")
    unique.write_bytes(b"unique-image-content")

    records = [
        _record(duplicate_b, "caption from first duplicate"),
        _record(duplicate_a, "caption from second duplicate"),
        _record(unique, "unique caption"),
    ]

    corpus, image_index = build_deduplicated_corpus(records=records, label_root=tmp_path)

    assert len(corpus) == 2
    assert corpus[0].image_path == str(duplicate_b)
    assert corpus[0].caption == "caption from first duplicate"

    duplicate_hash = corpus[0].image_id
    assert image_index[duplicate_hash].labels == {"对向来车", "低光照下，前车强逆光"}
    assert image_index[duplicate_hash].representative_path == str(duplicate_b)
    assert image_index[duplicate_hash].duplicate_count == 2


def test_evaluate_fixed_topk_hash_recall_counts_hash_matches_not_paths(tmp_path: Path) -> None:
    gt_dir = tmp_path / "雨天，前车刹车灯亮起"
    other_dir = tmp_path / "雨天"
    gt_dir.mkdir()
    other_dir.mkdir()
    shared_gt = gt_dir / "shared-gt.jpg"
    shared_other = other_dir / "shared-other.jpg"
    missed_gt = gt_dir / "missed.jpg"
    distractor = other_dir / "distractor.jpg"
    shared_gt.write_bytes(b"same-rain-brake")
    shared_other.write_bytes(b"same-rain-brake")
    missed_gt.write_bytes(b"missed-gt")
    distractor.write_bytes(b"distractor")

    records = [
        _record(shared_other, "shared duplicate chosen as representative"),
        _record(shared_gt, "same image in gt folder"),
        _record(missed_gt, "missed gt"),
        _record(distractor, "distractor"),
    ]
    corpus, image_index = build_deduplicated_corpus(records=records, label_root=tmp_path)
    hash_by_path = {meta.representative_path: image_hash for image_hash, meta in image_index.items()}
    gt_hashes = {
        image_hash
        for image_hash, meta in image_index.items()
        if "雨天，前车刹车灯亮起" in meta.labels
    }
    items = [
        _item(Path(corpus[0].image_path), 1),
        _item(Path(corpus[2].image_path), 2),
        _item(Path(corpus[1].image_path), 3),
    ]

    rows = evaluate_fixed_topk_hash_recall(
        items=items,
        gt_hashes=gt_hashes,
        top_ks=[1, 2],
        image_hash_by_path=hash_by_path,
    )

    assert rows[0]["top_k"] == 1
    assert rows[0]["recalled_gt_count"] == 1
    assert rows[0]["gt_count"] == 2
    assert rows[0]["recall"] == 0.5
    assert rows[1]["top_k"] == 2
    assert rows[1]["recalled_gt_count"] == 1
