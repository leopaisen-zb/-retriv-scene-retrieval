from __future__ import annotations

import pytest

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


@pytest.mark.parametrize(
    ("freeze_top_n", "final_top_k", "lexical_top_n", "message"),
    [
        (-1, 10, 3, "freeze_top_n 不能为负数"),
        (5, 0, 3, "final_top_k 必须为正整数"),
        (5, 10, 0, "lexical_top_n 必须为正整数"),
        (6, 5, 3, "freeze_top_n 不能大于 final_top_k"),
    ],
)
def test_rerank_rejects_invalid_validation_inputs(
    freeze_top_n: int,
    final_top_k: int,
    lexical_top_n: int,
    message: str,
) -> None:
    semantic = [_item(str(i)) for i in range(1, 11)]
    lexical = [_item("8"), _item("7"), _item("6")]

    with pytest.raises(ValueError, match=message):
        rerank_semantic_tail_with_lexical_support(
            semantic_items=semantic,
            lexical_items=lexical,
            freeze_top_n=freeze_top_n,
            final_top_k=final_top_k,
            lexical_top_n=lexical_top_n,
        )
