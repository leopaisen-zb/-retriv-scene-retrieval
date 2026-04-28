"""按图片内容去重后，严格评估 BM25 与 embedding 的固定 Top-k recall。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

from experiments.retrieval_percentile_recall_experiment import (
    DEFAULT_QUERY_LABEL_MAP,
    parse_query_file,
)
from pipelines.retrieval_pipeline import (
    BM25RetrievalBackend,
    CaptionRecord,
    Qwen3VLEmbeddingFaissBackend,
    RetrievalItem,
    load_caption_records,
)


@dataclass(frozen=True)
class DedupedImageMetadata:
    """去重后唯一图片内容的元信息。"""

    image_hash: str
    representative_path: str
    duplicate_paths: tuple[str, ...]
    labels: set[str]

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_paths)


def _resolved_path(value: str) -> str:
    return str(Path(value).resolve())


def sha256_file(path: str | Path) -> str:
    """计算文件内容 SHA256。"""
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"图片文件不存在：{file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_for_path(image_path: str, label_root: Path) -> str | None:
    try:
        rel = Path(image_path).resolve().relative_to(label_root.resolve())
    except ValueError:
        return None
    if not rel.parts:
        return None
    return rel.parts[0]


def build_deduplicated_corpus(
    records: list[CaptionRecord],
    label_root: str | Path,
) -> tuple[list[CaptionRecord], dict[str, DedupedImageMetadata]]:
    """将 caption 语料按图片内容 hash 去重，返回唯一语料和 hash 元信息。

    去重策略：同一 hash 只保留 caption JSON 中第一次出现的记录作为检索语料；
    所有重复路径都会合并到同一个 metadata，用于 GT 标签判断和结果追踪。
    """
    root = Path(label_root)
    selected_records: dict[str, CaptionRecord] = {}
    duplicate_paths: dict[str, list[str]] = {}
    labels_by_hash: dict[str, set[str]] = {}

    for record in records:
        image_path = _resolved_path(record.image_path)
        image_hash = sha256_file(image_path)
        duplicate_paths.setdefault(image_hash, []).append(image_path)
        label = _label_for_path(image_path=image_path, label_root=root)
        if label:
            labels_by_hash.setdefault(image_hash, set()).add(label)
        else:
            labels_by_hash.setdefault(image_hash, set())
        if image_hash not in selected_records:
            selected_records[image_hash] = CaptionRecord(
                image_id=image_hash,
                image_path=record.image_path,
                caption=record.caption,
                source_result_file=record.source_result_file,
            )

    corpus: list[CaptionRecord] = []
    image_index: dict[str, DedupedImageMetadata] = {}
    for image_hash, record in selected_records.items():
        paths = tuple(duplicate_paths[image_hash])
        representative_path = _resolved_path(record.image_path)
        corpus.append(
            CaptionRecord(
                image_id=image_hash,
                image_path=representative_path,
                caption=record.caption,
                source_result_file=record.source_result_file,
            )
        )
        image_index[image_hash] = DedupedImageMetadata(
            image_hash=image_hash,
            representative_path=representative_path,
            duplicate_paths=paths,
            labels=labels_by_hash[image_hash],
        )
    return corpus, image_index


def _parse_top_ks(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise ValueError("top-k 必须为正整数")
        values.append(value)
    if not values:
        raise ValueError("top-k 不能为空")
    return values


def _hash_lookup(image_hash_by_path: dict[str, str], image_path: str) -> str:
    found = image_hash_by_path.get(image_path)
    if found is not None:
        return found
    resolved = _resolved_path(image_path)
    found = image_hash_by_path.get(resolved)
    if found is None:
        raise ValueError(f"检索结果路径无法映射到图片 hash：{image_path}")
    return found


def evaluate_fixed_topk_hash_recall(
    items: list[RetrievalItem],
    gt_hashes: set[str],
    top_ks: list[int],
    image_hash_by_path: dict[str, str],
) -> list[dict[str, object]]:
    """基于唯一图片 hash 计算固定 Top-k recall。"""
    if not gt_hashes:
        raise ValueError("gt_hashes 不能为空")
    rows: list[dict[str, object]] = []
    for top_k in top_ks:
        candidate_items = items[:top_k]
        candidate_hashes = {
            _hash_lookup(image_hash_by_path=image_hash_by_path, image_path=item.image_path)
            for item in candidate_items
        }
        recalled = sorted(candidate_hashes & gt_hashes)
        missed = sorted(gt_hashes - candidate_hashes)
        rows.append(
            {
                "top_k": int(top_k),
                "candidate_count": len(candidate_items),
                "gt_count": len(gt_hashes),
                "recalled_gt_count": len(recalled),
                "recall": round(len(recalled) / len(gt_hashes), 6),
                "recalled_gt_hashes": recalled,
                "missed_gt_hashes": missed,
            }
        )
    return rows


def _format_ranked_items(
    items: list[RetrievalItem],
    gt_hashes: set[str],
    image_hash_by_path: dict[str, str],
    image_index: dict[str, DedupedImageMetadata],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, item in enumerate(items, start=1):
        image_hash = _hash_lookup(image_hash_by_path=image_hash_by_path, image_path=item.image_path)
        meta = image_index[image_hash]
        rows.append(
            {
                "rank": rank,
                "image_hash": image_hash,
                "image_path": meta.representative_path,
                "image_filename": Path(meta.representative_path).name,
                "score": item.score,
                "labels": sorted(meta.labels),
                "duplicate_count": meta.duplicate_count,
                "duplicate_paths": list(meta.duplicate_paths),
                "is_gt": image_hash in gt_hashes,
            }
        )
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "query",
                "backend",
                "top_k",
                "corpus_unique_count",
                "gt_count",
                "recalled_gt_count",
                "recall",
                "query_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按图片内容去重后严格对比 BM25 与 embedding recall")
    parser.add_argument("--caption-result-json", required=True, help="图片描述 JSON")
    parser.add_argument(
        "--query-file",
        default="input/processed/scene_queries_specific_en.json",
        help="沿用现有 query 文件",
    )
    parser.add_argument("--label-root", default="RetrInput_hh", help="人工标签图片根目录")
    parser.add_argument("--top-ks", default="10,20", help="逗号分隔的固定 Top-k")
    parser.add_argument("--output-json", required=True, help="评估 JSON 输出路径")
    parser.add_argument("--output-csv", required=True, help="评估 CSV 摘要输出路径")
    parser.add_argument(
        "--embedder-script-path",
        default="/home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py",
        help="Qwen3-VL-Embedding 官方 qwen3_vl_embedding.py 路径",
    )
    parser.add_argument(
        "--embedding-model-path",
        default="/home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B",
        help="Qwen3-VL-Embedding 模型路径",
    )
    parser.add_argument(
        "--embedding-instruction",
        default="Represent the user's input.",
        help="embedding 指令文本",
    )
    parser.add_argument(
        "--embedding-torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="embedding 模型 dtype",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    top_ks = _parse_top_ks(args.top_ks)
    records = load_caption_records(args.caption_result_json)
    if not records:
        raise ValueError("caption 语料为空")

    corpus, image_index = build_deduplicated_corpus(records=records, label_root=args.label_root)
    corpus_size = len(corpus)
    image_hash_by_path: dict[str, str] = {}
    for image_hash, meta in image_index.items():
        image_hash_by_path[meta.representative_path] = image_hash
        for duplicate_path in meta.duplicate_paths:
            image_hash_by_path[duplicate_path] = image_hash

    queries = parse_query_file(args.query_file)

    bm25_start = time.perf_counter()
    bm25_backend = BM25RetrievalBackend(records=corpus)
    bm25_build_seconds = time.perf_counter() - bm25_start

    embedding_start = time.perf_counter()
    embedding_backend = Qwen3VLEmbeddingFaissBackend(
        records=corpus,
        embedder_script_path=args.embedder_script_path,
        embedding_model_path=args.embedding_model_path,
        instruction=args.embedding_instruction,
        torch_dtype=args.embedding_torch_dtype,
        encode_batch_size=args.embedding_batch_size,
    )
    embedding_build_seconds = time.perf_counter() - embedding_start

    backends = {
        "bm25": (bm25_backend, bm25_build_seconds),
        "qwen3vl_embedding_faiss": (embedding_backend, embedding_build_seconds),
    }

    csv_rows: list[dict[str, object]] = []
    query_results: list[dict[str, object]] = []
    for query_row in queries:
        display_query = query_row["display_query"]
        search_query = query_row["search_query"]
        label = DEFAULT_QUERY_LABEL_MAP.get(display_query)
        if label is None:
            raise ValueError(f"缺少 query 到 GT 标签的映射：{display_query}")
        gt_hashes = {
            image_hash
            for image_hash, meta in image_index.items()
            if label in meta.labels
        }

        backend_results: list[dict[str, object]] = []
        for backend_name, (backend, build_seconds) in backends.items():
            query_start = time.perf_counter()
            items = backend.search(query=search_query, top_k=corpus_size)
            query_seconds = time.perf_counter() - query_start
            recall_rows = evaluate_fixed_topk_hash_recall(
                items=items,
                gt_hashes=gt_hashes,
                top_ks=top_ks,
                image_hash_by_path=image_hash_by_path,
            )
            for row in recall_rows:
                csv_rows.append(
                    {
                        "query": display_query,
                        "backend": backend_name,
                        "top_k": row["top_k"],
                        "corpus_unique_count": corpus_size,
                        "gt_count": row["gt_count"],
                        "recalled_gt_count": row["recalled_gt_count"],
                        "recall": row["recall"],
                        "query_seconds": round(query_seconds, 6),
                    }
                )
            backend_results.append(
                {
                    "backend": backend_name,
                    "build_seconds": round(build_seconds, 6),
                    "query_seconds": round(query_seconds, 6),
                    "recall_by_top_k": recall_rows,
                    "ranked_items": _format_ranked_items(
                        items=items,
                        gt_hashes=gt_hashes,
                        image_hash_by_path=image_hash_by_path,
                        image_index=image_index,
                    ),
                }
            )

        query_results.append(
            {
                "query": display_query,
                "search_query": search_query,
                "gt_label": label,
                "gt_count": len(gt_hashes),
                "gt_hashes": sorted(gt_hashes),
                "backends": backend_results,
            }
        )

    output = {
        "summary": {
            "caption_result_json": args.caption_result_json,
            "query_file": args.query_file,
            "label_root": args.label_root,
            "raw_caption_record_count": len(records),
            "deduplicated_corpus_count": corpus_size,
            "top_ks": top_ks,
            "backends": list(backends.keys()),
            "metric": "fixed_topk_hash_recall",
            "retrieval_scope": "all_deduplicated_images",
            "gt_usage": "gt_hashes_are_used_only_for_evaluation_not_for_retrieval_filtering",
        },
        "deduplicated_images": [
            {
                "image_hash": image_hash,
                "representative_path": meta.representative_path,
                "duplicate_paths": list(meta.duplicate_paths),
                "duplicate_count": meta.duplicate_count,
                "labels": sorted(meta.labels),
            }
            for image_hash, meta in image_index.items()
        ],
        "query_results": query_results,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(Path(args.output_csv), csv_rows)


if __name__ == "__main__":
    main()
