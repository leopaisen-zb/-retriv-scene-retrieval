from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_demo_module():
    module_path = Path("demo/scene_retrieval_demo/scene_retrieval_demo.py")
    spec = importlib.util.spec_from_file_location("scene_retrieval_demo", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_top_ks_accepts_csv_and_rejects_invalid_values() -> None:
    demo = _load_demo_module()

    assert demo.parse_top_ks("10,20,30") == [10, 20, 30]
    assert demo.parse_top_ks([5, 15]) == [5, 15]

    try:
        demo.parse_top_ks("10,0")
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("expected invalid top_k to fail")


def test_tokenize_bm25_keeps_english_numbers_and_chinese_terms() -> None:
    demo = _load_demo_module()

    assert demo.tokenize_bm25("Rainy road, CH4, 对向来车!") == [
        "rainy",
        "road",
        "ch4",
        "对向来车",
    ]


def test_evaluate_fixed_topk_hash_recall_counts_hash_matches() -> None:
    demo = _load_demo_module()
    items = [
        demo.RetrievalItem("h1", "/tmp/a.jpg", 3.0, "caption a", "captions.json"),
        demo.RetrievalItem("h2", "/tmp/b.jpg", 2.0, "caption b", "captions.json"),
        demo.RetrievalItem("h3", "/tmp/c.jpg", 1.0, "caption c", "captions.json"),
    ]
    path_to_hash = {
        "/tmp/a.jpg": "h1",
        "/tmp/b.jpg": "h2",
        "/tmp/c.jpg": "h3",
    }

    rows = demo.evaluate_fixed_topk_hash_recall(
        items=items,
        gt_hashes={"h2", "h3"},
        top_ks=[1, 2, 3],
        image_hash_by_path=path_to_hash,
    )

    assert rows[0]["recalled_gt_count"] == 0
    assert rows[1]["recalled_gt_count"] == 1
    assert rows[1]["recall"] == 0.5
    assert rows[2]["recalled_gt_count"] == 2
    assert rows[2]["recall"] == 1.0
