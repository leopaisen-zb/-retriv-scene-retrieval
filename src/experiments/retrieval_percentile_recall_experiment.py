"""按人工标签目录评估 embedding 与 BM25 的 top-percentage recall。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

from pipelines.retrieval_pipeline import (
    BM25RetrievalBackend,
    Qwen3VLEmbeddingFaissBackend,
    RetrievalItem,
    load_caption_records,
)


DEFAULT_QUERY_LABEL_MAP: dict[str, str] = {
    "vehicle ahead under strong backlighting in low-light conditions": "低光照下，前车强逆光",
    "oncoming vehicle approaching in the opposite lane": "对向来车",
    "rainy road scene with the brake lights of the vehicle ahead illuminated": "雨天，前车刹车灯亮起",
}


def parse_query_file(query_file: str) -> list[dict[str, str]]:
    """解析现有 query JSON，返回 display_query/search_query。"""
    path = Path(query_file)
    if not path.exists():
        raise ValueError(f"query 文件不存在：{query_file}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("queries", [])
    if not isinstance(items, list):
        raise ValueError("query 文件格式错误：queries 必须为列表")
    parsed: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parsed.append({"display_query": text, "search_query": text})
            continue
        if isinstance(item, dict):
            display_query = str(
                item.get("query") or item.get("display_query") or item.get("zh_query") or ""
            ).strip()
            search_query = str(
                item.get("english_query") or item.get("search_query") or item.get("en_query") or ""
            ).strip()
            if not display_query and not search_query:
                continue
            parsed.append(
                {
                    "display_query": display_query or search_query,
                    "search_query": search_query or display_query,
                }
            )
    if not parsed:
        raise ValueError("query 文件中未找到有效查询语句")
    return parsed


def percentile_candidate_count(total_count: int, percent: int) -> int:
    """将百分比阈值转换成候选数量，使用 ceil 并夹紧到语料规模内。"""
    if total_count <= 0:
        raise ValueError("total_count 必须为正整数")
    if percent <= 0:
        raise ValueError("percent 必须为正整数")
    return min(total_count, max(1, math.ceil(total_count * percent / 100.0)))


def _resolved_path(value: str) -> str:
    return str(Path(value).resolve())


def evaluate_percentile_recall(
    items: list[RetrievalItem],
    gt_paths: set[str],
    percentiles: Iterable[int],
    total_count: int,
) -> list[dict[str, object]]:
    """对同一条排序结果计算多个 top-percentage recall。"""
    normalized_gt = {_resolved_path(path) for path in gt_paths}
    if not normalized_gt:
        raise ValueError("gt_paths 不能为空")

    rows: list[dict[str, object]] = []
    for percent in percentiles:
        candidate_count = percentile_candidate_count(total_count=total_count, percent=int(percent))
        candidate_items = items[:candidate_count]
        candidate_paths = {_resolved_path(item.image_path) for item in candidate_items}
        recalled = sorted(normalized_gt & candidate_paths)
        missed = sorted(normalized_gt - candidate_paths)
        rows.append(
            {
                "percent": int(percent),
                "candidate_count": candidate_count,
                "gt_count": len(normalized_gt),
                "recalled_gt_count": len(recalled),
                "recall": round(len(recalled) / len(normalized_gt), 6),
                "recalled_gt_image_paths": recalled,
                "missed_gt_image_paths": missed,
            }
        )
    return rows


def _parse_percentiles(raw: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise ValueError("percentiles 必须为正整数")
        values.append(value)
    if not values:
        raise ValueError("percentiles 不能为空")
    return values


def _image_paths_under(label_root: Path, label: str) -> set[str]:
    label_dir = label_root / label
    if not label_dir.is_dir():
        raise ValueError(f"GT 标签目录不存在：{label_dir}")
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = {
        _resolved_path(str(path))
        for path in label_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }
    if not paths:
        raise ValueError(f"GT 标签目录没有图片：{label_dir}")
    return paths


def _format_ranked_items(items: list[RetrievalItem], gt_paths: set[str]) -> list[dict[str, object]]:
    normalized_gt = {_resolved_path(path) for path in gt_paths}
    rows: list[dict[str, object]] = []
    for rank, item in enumerate(items, start=1):
        image_path = _resolved_path(item.image_path)
        rows.append(
            {
                "rank": rank,
                "image_id": item.image_id,
                "image_path": image_path,
                "image_filename": Path(image_path).name,
                "score": item.score,
                "is_gt": image_path in normalized_gt,
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
                "percent",
                "candidate_count",
                "gt_count",
                "recalled_gt_count",
                "recall",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 top-percentage recall 对比 embedding 与 BM25")
    parser.add_argument("--caption-result-json", required=True, help="图片描述 JSON")
    parser.add_argument(
        "--query-file",
        default="input/processed/scene_queries_specific_en.json",
        help="沿用现有 query 文件",
    )
    parser.add_argument("--label-root", default="RetrInput_hh", help="人工标签图片根目录")
    parser.add_argument(
        "--percentiles",
        default="10,20,30,50",
        help="逗号分隔的 top 百分比阈值",
    )
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
    percentiles = _parse_percentiles(args.percentiles)
    label_root = Path(args.label_root)
    records = load_caption_records(args.caption_result_json)
    if not records:
        raise ValueError("caption 语料为空")
    corpus_size = len(records)

    queries = parse_query_file(args.query_file)
    semantic_backend = Qwen3VLEmbeddingFaissBackend(
        records=records,
        embedder_script_path=args.embedder_script_path,
        embedding_model_path=args.embedding_model_path,
        instruction=args.embedding_instruction,
        torch_dtype=args.embedding_torch_dtype,
        encode_batch_size=args.embedding_batch_size,
    )
    bm25_backend = BM25RetrievalBackend(records=records)
    backends = {
        "qwen3vl_embedding_faiss": semantic_backend,
        "bm25": bm25_backend,
    }

    result_rows: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    for query_row in queries:
        display_query = query_row["display_query"]
        search_query = query_row["search_query"]
        label = DEFAULT_QUERY_LABEL_MAP.get(display_query)
        if label is None:
            raise ValueError(f"缺少 query 到 GT 标签的映射：{display_query}")
        gt_paths = _image_paths_under(label_root=label_root, label=label)

        backend_rows: list[dict[str, object]] = []
        for backend_name, backend in backends.items():
            items = backend.search(query=search_query, top_k=corpus_size)
            recall_rows = evaluate_percentile_recall(
                items=items,
                gt_paths=gt_paths,
                percentiles=percentiles,
                total_count=corpus_size,
            )
            for row in recall_rows:
                csv_rows.append(
                    {
                        "query": display_query,
                        "backend": backend_name,
                        "percent": row["percent"],
                        "candidate_count": row["candidate_count"],
                        "gt_count": row["gt_count"],
                        "recalled_gt_count": row["recalled_gt_count"],
                        "recall": row["recall"],
                    }
                )
            backend_rows.append(
                {
                    "backend": backend_name,
                    "recall_by_percent": recall_rows,
                    "ranked_items": _format_ranked_items(items=items, gt_paths=gt_paths),
                }
            )

        result_rows.append(
            {
                "query": display_query,
                "search_query": search_query,
                "gt_label": label,
                "gt_count": len(gt_paths),
                "gt_image_paths": sorted(gt_paths),
                "backends": backend_rows,
            }
        )

    payload = {
        "summary": {
            "caption_result_json": args.caption_result_json,
            "query_file": args.query_file,
            "label_root": str(label_root),
            "corpus_image_count": corpus_size,
            "percentiles": percentiles,
            "backends": list(backends.keys()),
            "query_label_map": DEFAULT_QUERY_LABEL_MAP,
            "metric": "top_percentage_recall",
            "candidate_count_rule": "ceil(corpus_image_count * percent / 100)",
        },
        "query_results": result_rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(Path(args.output_csv), csv_rows)


if __name__ == "__main__":
    main()
