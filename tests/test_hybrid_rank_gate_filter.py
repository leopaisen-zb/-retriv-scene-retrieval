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


@pytest.mark.parametrize(
    ("candidate_top_k", "semantic_top_n", "lexical_top_n", "output_max_k"),
    [
        (20.5, 10, 8, 10),
        ("20", 10, 8, 10),
        (20, 10.0, 8, 10),
        (20, 10, "8", 10),
        (20, 10, 8, 10.0),
    ],
)
def test_filter_rejects_non_integer_gate_bounds(
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
