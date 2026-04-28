from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from experiments.frozen_embedding_hybrid_filter_rank_gate_experiment import (
    build_filter_manifest,
    main,
    parse_args,
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


def test_write_query_filter_comparison_dirs_creates_expected_subfolders(
    tmp_path: Path,
) -> None:
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


def test_write_query_filter_comparison_dirs_truncates_bm25_to_top8_contract(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(1, 11):
        (image_dir / f"{index:02d}.jpg").write_bytes(b"fake image")

    embedding_row = _row(
        "夜间，前车灯组清晰可见",
        [
            (1, "e1", str(image_dir / "01.jpg")),
            (2, "e2", str(image_dir / "02.jpg")),
            (3, "e3", str(image_dir / "03.jpg")),
            (4, "e4", str(image_dir / "04.jpg")),
        ],
    )
    bm25_row = _row(
        "夜间，前车灯组清晰可见",
        [
            (1, "b1", str(image_dir / "01.jpg")),
            (2, "b2", str(image_dir / "02.jpg")),
            (3, "b3", str(image_dir / "03.jpg")),
            (4, "b4", str(image_dir / "04.jpg")),
            (5, "b5", str(image_dir / "05.jpg")),
            (6, "b6", str(image_dir / "06.jpg")),
            (7, "b7", str(image_dir / "07.jpg")),
            (8, "b8", str(image_dir / "08.jpg")),
            (9, "b9", str(image_dir / "09.jpg")),
            (10, "b10", str(image_dir / "10.jpg")),
        ],
    )
    filtered_row = _row(
        "夜间，前车灯组清晰可见",
        [
            (1, "b1", str(image_dir / "01.jpg")),
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
    manifest = json.loads((query_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["bm25_top8_image_ids"] == [
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "b8",
    ]
    copied_files = sorted(
        path.name for path in (query_path / "bm25_top8_candidates").iterdir()
    )
    assert copied_files == [
        "01_01.jpg",
        "02_02.jpg",
        "03_03.jpg",
        "04_04.jpg",
        "05_05.jpg",
        "06_06.jpg",
        "07_07.jpg",
        "08_08.jpg",
    ]


def test_write_query_filter_comparison_dirs_handles_empty_filtered_rows(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (image_dir / name).write_bytes(b"fake image")

    embedding_row = _row(
        "雾天，前车远灯可见",
        [
            (1, "a", str(image_dir / "a.jpg")),
            (2, "b", str(image_dir / "b.jpg")),
            (3, "c", str(image_dir / "c.jpg")),
        ],
    )
    bm25_row = _row(
        "雾天，前车远灯可见",
        [
            (1, "c", str(image_dir / "c.jpg")),
        ],
    )
    filtered_row = {"query": "雾天，前车远灯可见", "items": []}

    query_dir = write_query_filter_comparison_dirs(
        base_dir=tmp_path / "comparisons",
        embedding_row=embedding_row,
        bm25_row=bm25_row,
        filtered_row=filtered_row,
        candidate_top_k=3,
        semantic_top_n=2,
        semantic_report_path="semantic.json",
        filtered_report_path="filtered.json",
    )

    query_path = Path(query_dir)
    assert (query_path / "embedding_top20").is_dir()
    assert (query_path / "embedding_top10").is_dir()
    assert (query_path / "bm25_top8_candidates").is_dir()
    assert (query_path / "final_filtered").is_dir()
    assert (query_path / "filtered_out_from_embedding_top10").is_dir()

    manifest = json.loads((query_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["final_result_count"] == 0
    assert manifest["final_filtered_image_ids"] == []
    assert manifest["filtered_out_image_ids"] == ["a", "b"]


def test_parse_args_help_marks_fixed_contract_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["prog", "--help"],
    )

    with pytest.raises(SystemExit) as excinfo:
        parse_args()

    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "--candidate-top-k" in captured.out
    assert "Fixed contract value: 20." in captured.out
    assert "Override will raise at" in captured.out


@pytest.mark.parametrize(
    ("candidate_top_k", "semantic_top_n", "lexical_top_n", "output_max_k", "message"),
    [
        (19, 10, 8, 10, "candidate_top_k 必须固定为 20"),
        (20, 9, 8, 10, "semantic_top_n 必须固定为 10"),
        (20, 10, 7, 10, "lexical_top_n 必须固定为 8"),
        (20, 10, 8, 9, "output_max_k 必须固定为 10"),
    ],
)
def test_main_rejects_non_default_rank_gate_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_top_k: int,
    semantic_top_n: int,
    lexical_top_n: int,
    output_max_k: int,
    message: str,
) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake image")

    semantic_report = tmp_path / "semantic.json"
    semantic_report.write_text(
        json.dumps(
            {
                "query_results": [
                    {
                        "query": "雨天，前车刹车灯亮起",
                        "search_query": "雨天 前车 刹车灯 亮起",
                        "items": [
                            {
                                "rank": 1,
                                "image_id": "a",
                                "image_path": str(image_path),
                                "score": 0.9,
                                "caption": "caption for a",
                                "source_result_file": "semantic.json",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    caption_json = tmp_path / "caption.json"
    caption_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "success": True,
                        "caption": "caption for a",
                        "image_path": str(image_path),
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    args = argparse.Namespace(
        semantic_report=str(semantic_report),
        caption_result_json=str(caption_json),
        bm25_view_json=str(tmp_path / "bm25_view.json"),
        output_json=str(tmp_path / "output.json"),
        output_csv=str(tmp_path / "output.csv"),
        filtered_output_dir=str(tmp_path / "filtered"),
        comparison_dir=str(tmp_path / "comparisons"),
        candidate_top_k=candidate_top_k,
        semantic_top_n=semantic_top_n,
        lexical_top_n=lexical_top_n,
        output_max_k=output_max_k,
    )
    monkeypatch.setattr(
        "experiments.frozen_embedding_hybrid_filter_rank_gate_experiment.parse_args",
        lambda: args,
    )

    with pytest.raises(ValueError, match=message):
        main()
