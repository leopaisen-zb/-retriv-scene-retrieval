"""文本检索图片流水线实现。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

try:
    import faiss
except Exception as exc:  # pylint: disable=broad-except
    faiss = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as exc:  # pylint: disable=broad-except
    TfidfVectorizer = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None

try:
    from rank_bm25 import BM25Okapi
except Exception as exc:  # pylint: disable=broad-except
    BM25Okapi = None  # type: ignore[assignment]
    _BM25_IMPORT_ERROR = exc
else:
    _BM25_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)

_INPUT_RAW_MARKER = "/input/raw/"


@dataclass(frozen=True)
class RetrievalItem:
    """检索结果项。"""

    image_id: str
    image_path: str
    score: float
    caption: str
    source_result_file: str


@dataclass(frozen=True)
class CaptionRecord:
    """描述语料记录。"""

    image_id: str
    image_path: str
    caption: str
    source_result_file: str


class RetrievalBackend(Protocol):
    """检索后端协议。"""

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        """执行文本检索。

        Args:
            query: 检索文本。
            top_k: 返回数量。

        Returns:
            list[RetrievalItem]: 检索结果列表。
        """


class TfidfFaissRetrievalBackend:
    """基于 TF-IDF 向量与 FAISS 内积检索的后端。

    说明：
        - 当前语料规模较小时，TF-IDF 能快速建立稳定向量检索基线；
        - 后续可替换为更强文本 embedding（如 e5/bge/qwen embedding）而不改上层接口。
    """

    vectorizer: TfidfVectorizer
    index: faiss.IndexFlatIP
    records: list[CaptionRecord]

    def __init__(self, records: list[CaptionRecord], min_df: int = 1) -> None:
        """初始化 TF-IDF 与 FAISS 索引。

        Args:
            records: 描述语料记录。
            min_df: TF-IDF 最小文档频次阈值。

        Raises:
            ValueError: 输入语料为空。
            RuntimeError: 索引构建失败。
        """
        if faiss is None:
            raise RuntimeError(f"缺少依赖 faiss，请先安装。原始错误：{_FAISS_IMPORT_ERROR}")
        if TfidfVectorizer is None:
            raise RuntimeError(f"缺少依赖 scikit-learn，请先安装。原始错误：{_SKLEARN_IMPORT_ERROR}")

        if not records:
            raise ValueError("检索语料为空，无法构建索引")
        self.records = records
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            min_df=min_df,
            strip_accents="unicode",
            ngram_range=(1, 2),
        )
        try:
            matrix = self.vectorizer.fit_transform([record.caption for record in records]).astype("float32")
            dense_vectors = matrix.toarray()
            dim = dense_vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(dense_vectors)
            LOGGER.info("FAISS index built with %d records", len(records))
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"FAISS 索引构建失败：{exc}") from exc

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        """执行向量检索。

        Args:
            query: 检索文本。
            top_k: 返回数量。

        Returns:
            list[RetrievalItem]: 检索结果列表。

        Raises:
            ValueError: 参数非法。
            RuntimeError: 检索失败。
        """
        if not query.strip():
            raise ValueError("检索文本不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        try:
            query_vector = self.vectorizer.transform([query]).astype("float32").toarray()
            search_k = min(top_k, len(self.records))
            scores, indices = self.index.search(query_vector, search_k)

            items: list[RetrievalItem] = []
            for score, idx in zip(scores[0], indices[0], strict=True):
                if idx < 0:
                    continue
                record = self.records[idx]
                items.append(
                    RetrievalItem(
                        image_id=record.image_id,
                        image_path=record.image_path,
                        score=float(score),
                        caption=record.caption,
                        source_result_file=record.source_result_file,
                    )
                )
            return items
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"向量检索失败：{exc}") from exc


class Qwen3VLEmbeddingFaissBackend:
    """基于 Qwen3-VL-Embedding 与 FAISS 内积检索的后端。"""

    index: faiss.IndexFlatIP
    records: list[CaptionRecord]
    embedder: Any
    instruction: str

    def __init__(
        self,
        records: list[CaptionRecord],
        embedder_script_path: str,
        embedding_model_path: str,
        instruction: str = "Represent the user's input.",
        torch_dtype: str = "bfloat16",
        encode_batch_size: int = 8,
    ) -> None:
        """初始化 Qwen3-VL-Embedding 检索后端。

        Args:
            records: 描述语料记录。
            embedder_script_path: Qwen3-VL-Embedding 官方 Python 模块路径。
            embedding_model_path: embedding 模型路径。
            instruction: embedding 指令文本。
            torch_dtype: 模型加载 dtype（float16/bfloat16/float32）。
            encode_batch_size: 文本编码批大小，降低显存峰值用。

        Raises:
            ValueError: 输入参数不合法。
            RuntimeError: 模型加载或索引构建失败。
        """
        if faiss is None:
            raise RuntimeError(f"缺少依赖 faiss，请先安装。原始错误：{_FAISS_IMPORT_ERROR}")
        if not records:
            raise ValueError("检索语料为空，无法构建索引")

        self.records = records
        self.embedder_script_path = str(Path(embedder_script_path))
        self.embedding_model_path = str(Path(embedding_model_path))
        self.instruction = instruction
        self.encode_batch_size = max(1, int(encode_batch_size))
        self.embedder = self._load_embedder(
            embedder_script_path=self.embedder_script_path,
            embedding_model_path=self.embedding_model_path,
            torch_dtype=torch_dtype,
        )
        vectors = self._encode_texts(texts=[record.caption for record in records])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        LOGGER.info("Qwen3-VL-Embedding FAISS index built with %d records", len(records))

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: str) -> torch.dtype:
        """将字符串解析为 torch.dtype。

        Args:
            torch_dtype: dtype 字符串。

        Returns:
            torch.dtype: torch 数据类型。
        """
        mapping: dict[str, torch.dtype] = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(torch_dtype, torch.bfloat16)

    @staticmethod
    def _load_embedder(
        embedder_script_path: str,
        embedding_model_path: str,
        torch_dtype: str,
    ) -> Any:
        """动态加载 Qwen3VLEmbedder 实现。

        Args:
            embedder_script_path: Qwen3-VL-Embedding 官方 Python 模块路径。
            embedding_model_path: embedding 模型路径。
            torch_dtype: 模型 dtype。

        Returns:
            Any: Qwen3VLEmbedder 实例。

        Raises:
            RuntimeError: 加载失败。
        """
        script_path = Path(embedder_script_path)
        if not script_path.exists():
            raise RuntimeError(f"Embedding 脚本不存在：{embedder_script_path}")
        if not Path(embedding_model_path).exists():
            raise RuntimeError(f"Embedding 模型路径不存在：{embedding_model_path}")

        try:
            spec = importlib.util.spec_from_file_location("qwen3vl_embedding_scripts", str(script_path))
            if spec is None or spec.loader is None:
                raise RuntimeError("无法加载 embedding 脚本 spec")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            embedder_cls = getattr(module, "Qwen3VLEmbedder")
            return embedder_cls(
                model_name_or_path=embedding_model_path,
                torch_dtype=Qwen3VLEmbeddingFaissBackend._resolve_torch_dtype(torch_dtype),
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"Qwen3VLEmbedder 初始化失败：{exc}") from exc

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """对文本列表生成归一化向量。

        Args:
            texts: 文本列表。

        Returns:
            np.ndarray: float32 向量矩阵。

        Raises:
            RuntimeError: 编码失败。
        """
        try:
            chunks: list[np.ndarray] = []
            for start in range(0, len(texts), self.encode_batch_size):
                part = texts[start : start + self.encode_batch_size]
                model_inputs = [{"text": text, "instruction": self.instruction} for text in part]
                embeddings = self.embedder.process(model_inputs, normalize=True)
                if hasattr(embeddings, "detach"):
                    embeddings = embeddings.float().detach().cpu().numpy()
                vectors = np.asarray(embeddings, dtype=np.float32)
                if vectors.ndim != 2:
                    raise RuntimeError("embedding 结果维度错误，期望二维矩阵")
                chunks.append(vectors)
            return np.concatenate(chunks, axis=0)
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"文本向量编码失败：{exc}") from exc

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        """执行向量检索。

        Args:
            query: 检索文本。
            top_k: 返回数量。

        Returns:
            list[RetrievalItem]: 检索结果列表。
        """
        if not query.strip():
            raise ValueError("检索文本不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        query_vector = self._encode_texts(texts=[query])
        search_k = min(top_k, len(self.records))
        scores, indices = self.index.search(query_vector, search_k)

        items: list[RetrievalItem] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            record = self.records[idx]
            items.append(
                RetrievalItem(
                    image_id=record.image_id,
                    image_path=record.image_path,
                    score=float(score),
                    caption=record.caption,
                    source_result_file=record.source_result_file,
                )
            )
        return items


def _tokenize_bm25(text: str) -> list[str]:
    """为 BM25 做轻量分词，优先适配英文 caption/query。"""
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())


class BM25RetrievalBackend:
    """基于 rank_bm25 的词法检索后端。"""

    records: list[CaptionRecord]
    tokenized_corpus: list[list[str]]
    bm25: Any

    def __init__(self, records: list[CaptionRecord]) -> None:
        if BM25Okapi is None:
            raise RuntimeError(f"缺少依赖 rank_bm25，请先安装。原始错误：{_BM25_IMPORT_ERROR}")
        if not records:
            raise ValueError("检索语料为空，无法构建索引")

        self.records = records
        self.tokenized_corpus = [_tokenize_bm25(record.caption) for record in records]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        if not query.strip():
            raise ValueError("检索文本不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        query_tokens = _tokenize_bm25(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[: min(top_k, len(self.records))]
        items: list[RetrievalItem] = []
        for idx, score in ranked:
            record = self.records[idx]
            items.append(
                RetrievalItem(
                    image_id=record.image_id,
                    image_path=record.image_path,
                    score=float(score),
                    caption=record.caption,
                    source_result_file=record.source_result_file,
                )
            )
        return items


class HybridRetrievalBackend:
    """并行调用语义检索与词法检索，并使用 RRF 合并排序。"""

    semantic_backend: RetrievalBackend
    lexical_backend: RetrievalBackend
    rrf_k: int

    def __init__(
        self,
        semantic_backend: RetrievalBackend,
        lexical_backend: RetrievalBackend,
        rrf_k: int = 60,
    ) -> None:
        self.semantic_backend = semantic_backend
        self.lexical_backend = lexical_backend
        self.rrf_k = max(0, int(rrf_k))

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        if not query.strip():
            raise ValueError("检索文本不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")

        semantic_items = self.semantic_backend.search(query=query, top_k=top_k)
        lexical_items = self.lexical_backend.search(query=query, top_k=top_k)
        fused: dict[str, tuple[float, RetrievalItem]] = {}

        for items in (semantic_items, lexical_items):
            for rank, item in enumerate(items, start=1):
                fused_score = 1.0 / (self.rrf_k + rank)
                prev = fused.get(item.image_id)
                if prev is None:
                    fused[item.image_id] = (fused_score, item)
                    continue
                fused[item.image_id] = (prev[0] + fused_score, prev[1])

        merged = sorted(fused.values(), key=lambda value: value[0], reverse=True)[:top_k]
        return [
            RetrievalItem(
                image_id=item.image_id,
                image_path=item.image_path,
                score=score,
                caption=item.caption,
                source_result_file=item.source_result_file,
            )
            for score, item in merged
        ]


class RetrievalPipeline:
    """文本到图片检索流水线。"""

    backend: RetrievalBackend

    def __init__(self, backend: RetrievalBackend) -> None:
        """初始化检索流水线。

        Args:
            backend: 检索后端实例。
        """
        self.backend = backend

    def search(self, query: str, top_k: int = 5) -> list[RetrievalItem]:
        """执行检索并返回结果。

        Args:
            query: 检索文本描述。
            top_k: 返回数量上限。

        Returns:
            list[RetrievalItem]: 按相关性排序的结果。

        Raises:
            ValueError: 输入参数非法。
            RuntimeError: 检索执行失败。
        """
        if not query.strip():
            raise ValueError("检索文本不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数")
        try:
            return self.backend.search(query=query, top_k=top_k)
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"文本检索执行失败：{exc}") from exc


def _resolve_migrated_image_path(image_path: str, project_root: Path | None = None) -> str:
    """将迁移前的绝对图片路径映射到当前 Retriv 工作区。

    Args:
        image_path: caption 结果中记录的图片路径。
        project_root: Retriv 项目根路径；测试时可注入。

    Returns:
        str: 可用于读取图片的路径。若原路径存在，则原样返回。
    """
    path = Path(image_path)
    if path.exists():
        return image_path

    normalized = image_path.replace("\\", "/")
    if _INPUT_RAW_MARKER not in normalized:
        return image_path

    relative_under_raw = normalized.split(_INPUT_RAW_MARKER, 1)[1]
    root = project_root if project_root is not None else Path(__file__).resolve().parents[2]
    migrated = root / "input" / "raw" / relative_under_raw
    if migrated.exists():
        return str(migrated)
    return image_path


def merge_multi_query_retrieval(
    backend: Any,
    queries: list[str],
    top_k: int,
    corpus_size: int,
) -> list[RetrievalItem]:
    """多条英文 query 各自检索后，按图片路径取最大相似度，再全局取 top_k。

    Args:
        backend: 实现 ``search(query, top_k)`` 的检索后端。
        queries: 英文 query 列表（翻译 + 相近场景）。
        top_k: 最终返回条数。
        corpus_size: 语料条数，用于单次检索拉全量排序。

    Returns:
        list[RetrievalItem]: 合并后的结果，分数为各 query 上的最大相似度。

    Raises:
        ValueError: 参数非法。
    """
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")
    if corpus_size <= 0:
        return []
    search_k = max(1, corpus_size)
    best: dict[str, tuple[float, RetrievalItem]] = {}
    for q in queries:
        q = q.strip()
        if not q:
            continue
        items = backend.search(query=q, top_k=search_k)
        for it in items:
            prev = best.get(it.image_path)
            if prev is None or it.score > prev[0]:
                best[it.image_path] = (it.score, it)
    merged = sorted(best.values(), key=lambda x: x[0], reverse=True)[:top_k]
    return [
        RetrievalItem(
            image_id=item.image_id,
            image_path=item.image_path,
            score=score,
            caption=item.caption,
            source_result_file=item.source_result_file,
        )
        for score, item in merged
    ]


def load_caption_records(result_json_path: str) -> list[CaptionRecord]:
    """从图片描述结果 JSON 加载语料记录。

    Args:
        result_json_path: 描述结果 JSON 路径。

    Returns:
        list[CaptionRecord]: 语料记录列表。

    Raises:
        ValueError: 路径非法或文件内容异常。
    """
    path = Path(result_json_path)
    if not path.exists():
        raise ValueError(f"描述结果文件不存在：{result_json_path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("描述结果格式非法：results 必须为列表")

    records: list[CaptionRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if not row.get("success", False):
            continue
        caption = str(row.get("caption", "")).strip()
        image_path = _resolve_migrated_image_path(str(row.get("image_path", "")).strip())
        if not caption or not image_path:
            continue
        image_id = Path(image_path).stem
        records.append(
            CaptionRecord(
                image_id=image_id,
                image_path=image_path,
                caption=caption,
                source_result_file=str(path),
            )
        )
    return records
