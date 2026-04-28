from __future__ import annotations

from pipelines.retrieval_pipeline import RetrievalItem


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} 必须为正整数")
    if value <= 0:
        raise ValueError(f"{name} 必须为正整数")


def filter_semantic_candidates_by_rank_gates(
    semantic_items: list[RetrievalItem],
    lexical_items: list[RetrievalItem],
    candidate_top_k: int = 20,
    semantic_top_n: int = 10,
    lexical_top_n: int = 8,
    output_max_k: int = 10,
) -> list[RetrievalItem]:
    _require_positive_int("candidate_top_k", candidate_top_k)
    _require_positive_int("semantic_top_n", semantic_top_n)
    _require_positive_int("lexical_top_n", lexical_top_n)
    _require_positive_int("output_max_k", output_max_k)
    if semantic_top_n > candidate_top_k:
        raise ValueError("semantic_top_n 不能大于 candidate_top_k")

    semantic_candidates = semantic_items[:candidate_top_k]
    semantic_gate = semantic_candidates[:semantic_top_n]
    lexical_gate_ids = {item.image_id for item in lexical_items[:lexical_top_n]}

    surviving = [item for item in semantic_gate if item.image_id in lexical_gate_ids]
    return surviving[:output_max_k]
