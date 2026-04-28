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
    if freeze_top_n > final_top_k:
        raise ValueError("freeze_top_n 不能大于 final_top_k")

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
