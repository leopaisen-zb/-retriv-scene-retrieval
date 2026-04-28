# BM25 And Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real `BM25RetrievalBackend` and a hybrid embedding+BM25 backend while keeping the existing caption-JSON-to-output pipeline unchanged.

**Architecture:** Keep `RetrievalPipeline` and output formatting untouched. Extend `src/pipelines/retrieval_pipeline.py` with a lexical BM25 backend plus a fusion backend that combines semantic and lexical results with RRF, then expose both through the CLI backend selector.

**Tech Stack:** Python, `rank_bm25`, existing FAISS/transformers backends, `pytest`

---

### Task 1: Lock BM25 backend behavior with tests

**Files:**
- Create: `tests/test_bm25_hybrid_retrieval.py`
- Modify: `src/pipelines/retrieval_pipeline.py`

- [ ] **Step 1: Write the failing BM25 tests**

```python
from pipelines.retrieval_pipeline import BM25RetrievalBackend, CaptionRecord


def test_bm25_backend_ranks_exact_term_overlap_highest():
    records = [
        CaptionRecord("a", "/tmp/a.jpg", "vehicle ahead under strong backlighting", "captions.json"),
        CaptionRecord("b", "/tmp/b.jpg", "rainy road with brake lights ahead", "captions.json"),
    ]

    backend = BM25RetrievalBackend(records=records)

    items = backend.search("strong backlighting vehicle ahead", top_k=2)

    assert [item.image_id for item in items] == ["a", "b"]


def test_bm25_backend_rejects_blank_query():
    backend = BM25RetrievalBackend(
        records=[CaptionRecord("a", "/tmp/a.jpg", "caption text", "captions.json")]
    )

    with pytest.raises(ValueError, match="检索文本不能为空"):
        backend.search("   ", top_k=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_bm25_hybrid_retrieval.py`
Expected: FAIL because `BM25RetrievalBackend` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class BM25RetrievalBackend:
    def __init__(self, records: list[CaptionRecord]) -> None:
        ...

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_bm25_hybrid_retrieval.py`
Expected: PASS for the BM25 backend tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_bm25_hybrid_retrieval.py src/pipelines/retrieval_pipeline.py
git commit -m "feat: add bm25 retrieval backend"
```

### Task 2: Lock hybrid fusion behavior with tests

**Files:**
- Modify: `tests/test_bm25_hybrid_retrieval.py`
- Modify: `src/pipelines/retrieval_pipeline.py`

- [ ] **Step 1: Write the failing hybrid tests**

```python
def test_hybrid_backend_merges_semantic_and_lexical_results_with_rrf():
    semantic = FakeBackend([
        RetrievalItem("a", "/tmp/a.jpg", 0.9, "caption a", "captions.json"),
        RetrievalItem("b", "/tmp/b.jpg", 0.8, "caption b", "captions.json"),
    ])
    lexical = FakeBackend([
        RetrievalItem("b", "/tmp/b.jpg", 3.5, "caption b", "captions.json"),
        RetrievalItem("c", "/tmp/c.jpg", 3.1, "caption c", "captions.json"),
    ])

    backend = HybridRetrievalBackend(semantic_backend=semantic, lexical_backend=lexical)

    items = backend.search("query", top_k=3)

    assert [item.image_id for item in items] == ["b", "a", "c"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_bm25_hybrid_retrieval.py`
Expected: FAIL because `HybridRetrievalBackend` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
class HybridRetrievalBackend:
    def __init__(self, semantic_backend: RetrievalBackend, lexical_backend: RetrievalBackend, rrf_k: int = 60) -> None:
        ...

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_bm25_hybrid_retrieval.py`
Expected: PASS for the hybrid fusion tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_bm25_hybrid_retrieval.py src/pipelines/retrieval_pipeline.py
git commit -m "feat: add hybrid retrieval backend"
```

### Task 3: Expose BM25 and hybrid through the CLI

**Files:**
- Modify: `src/experiments/text_to_image_retrieval_experiment.py`
- Test: `tests/test_bm25_hybrid_retrieval.py`

- [ ] **Step 1: Write the failing CLI selection test**

```python
def test_cli_backend_choices_include_bm25_and_hybrid():
    source = Path("src/experiments/text_to_image_retrieval_experiment.py").read_text(encoding="utf-8")
    assert '"bm25"' in source
    assert '"hybrid_embedding_bm25"' in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_bm25_hybrid_retrieval.py`
Expected: FAIL because the new backend choices are not wired yet.

- [ ] **Step 3: Write minimal implementation**

```python
parser.add_argument(
    "--backend",
    choices=["qwen3vl_embedding_faiss", "tfidf_faiss", "bm25", "hybrid_embedding_bm25"],
    ...
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m pytest -q tests/test_bm25_hybrid_retrieval.py`
Expected: PASS for backend-choice coverage.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/text_to_image_retrieval_experiment.py tests/test_bm25_hybrid_retrieval.py
git commit -m "feat: expose bm25 and hybrid retrieval cli options"
```
