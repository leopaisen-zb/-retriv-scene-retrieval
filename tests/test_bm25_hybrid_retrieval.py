from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


if "torch" not in sys.modules:
    sys.modules["torch"] = types.SimpleNamespace(
        dtype=object,
        float16="float16",
        bfloat16="bfloat16",
        float32="float32",
    )

from pipelines import retrieval_pipeline as rp


class FakeBM25Okapi:
    def __init__(self, corpus: list[list[str]]) -> None:
        self.corpus = corpus

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        query_terms = set(query_tokens)
        for doc_tokens in self.corpus:
            overlap = sum(1 for token in doc_tokens if token in query_terms)
            scores.append(float(overlap))
        return scores


class FakeBackend:
    def __init__(self, items: list[rp.RetrievalItem]) -> None:
        self.items = items

    def search(self, query: str, top_k: int) -> list[rp.RetrievalItem]:
        assert query == "test query"
        return self.items[:top_k]


def test_bm25_backend_ranks_exact_term_overlap_highest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "BM25Okapi", FakeBM25Okapi, raising=False)

    records = [
        rp.CaptionRecord(
            image_id="a",
            image_path="/tmp/a.jpg",
            caption="vehicle ahead under strong backlighting",
            source_result_file="captions.json",
        ),
        rp.CaptionRecord(
            image_id="b",
            image_path="/tmp/b.jpg",
            caption="rainy road with brake lights ahead",
            source_result_file="captions.json",
        ),
    ]

    backend = rp.BM25RetrievalBackend(records=records)

    items = backend.search("strong backlighting vehicle ahead", top_k=2)

    assert [item.image_id for item in items] == ["a", "b"]
    assert items[0].score > items[1].score


def test_bm25_backend_rejects_blank_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rp, "BM25Okapi", FakeBM25Okapi, raising=False)

    backend = rp.BM25RetrievalBackend(
        records=[
            rp.CaptionRecord(
                image_id="a",
                image_path="/tmp/a.jpg",
                caption="caption text",
                source_result_file="captions.json",
            )
        ]
    )

    with pytest.raises(ValueError, match="检索文本不能为空"):
        backend.search("   ", top_k=1)


def test_hybrid_backend_merges_semantic_and_lexical_results_with_rrf() -> None:
    semantic = FakeBackend(
        [
            rp.RetrievalItem("a", "/tmp/a.jpg", 0.9, "caption a", "captions.json"),
            rp.RetrievalItem("b", "/tmp/b.jpg", 0.8, "caption b", "captions.json"),
        ]
    )
    lexical = FakeBackend(
        [
            rp.RetrievalItem("b", "/tmp/b.jpg", 3.5, "caption b", "captions.json"),
            rp.RetrievalItem("c", "/tmp/c.jpg", 3.1, "caption c", "captions.json"),
        ]
    )

    backend = rp.HybridRetrievalBackend(
        semantic_backend=semantic,
        lexical_backend=lexical,
        rrf_k=0,
    )

    items = backend.search("test query", top_k=3)

    assert [item.image_id for item in items] == ["b", "a", "c"]


def test_cli_backend_choices_include_bm25_and_hybrid() -> None:
    source = Path("src/experiments/text_to_image_retrieval_experiment.py").read_text(encoding="utf-8")

    assert '"bm25"' in source
    assert '"hybrid_embedding_bm25"' in source
