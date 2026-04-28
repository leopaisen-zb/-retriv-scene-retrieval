from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.text_to_image_retrieval_experiment import (
    _format_items,
    _organize_images_by_query,
    _sanitize_query_dir_name,
    _to_json_payload,
    _write_recall_summary_csv,
)
from pipelines.hybrid_tail_rerank import rerank_semantic_tail_with_lexical_support
from pipelines.retrieval_pipeline import (
    BM25RetrievalBackend,
    CaptionRecord,
    RetrievalItem,
    load_caption_records,
)
from utils.bm25_lexical_view import build_bm25_view_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结 embedding 基线的 hybrid 主实验")
    parser.add_argument(
        "--semantic-report",
        default="output/reports/retrieval_issues_all_qwen3vl_embedding_specific_en_no_threshold.json",
    )
    parser.add_argument(
        "--caption-result-json",
        default="output/reports/caption_issues_all_qwen35.json",
    )
    parser.add_argument(
        "--bm25-view-json",
        default="output/artifacts/bm25_views/hybrid_main_experiment.json",
    )
    parser.add_argument(
        "--output-json",
        default="output/reports/retrieval_issues_all_hybrid_main_experiment.json",
    )
    parser.add_argument(
        "--output-csv",
        default="output/reports/retrieval_issues_all_hybrid_main_experiment_recall_summary.csv",
    )
    parser.add_argument(
        "--hybrid-output-dir",
        default="output/retrieval_output_hybrid_main_experiment",
    )
    parser.add_argument(
        "--comparison-dir",
        default="output/comparisons/hybrid_main_experiment",
    )
    parser.add_argument("--freeze-top-n", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--lexical-top-n", type=int, default=10)
    return parser.parse_args()


def _semantic_items_from_row(row: dict[str, object]) -> list[RetrievalItem]:
    return [
        RetrievalItem(
            image_id=str(item["image_id"]),
            image_path=str(item["image_path"]),
            score=float(item["score"]),
            caption=str(item["caption"]),
            source_result_file=str(item["source_result_file"]),
        )
        for item in row["items"]
    ]


def _lexical_caption_records(
    caption_records: list[CaptionRecord],
) -> tuple[list[CaptionRecord], list[dict[str, object]]]:
    lexical_rows = build_bm25_view_rows(caption_records)
    lexical_records = [
        CaptionRecord(
            image_id=row.image_id,
            image_path=row.image_path,
            caption=row.bm25_text,
            source_result_file="bm25_lexical_view",
        )
        for row in lexical_rows
    ]
    lexical_payload = [
        {
            "image_id": row.image_id,
            "image_path": row.image_path,
            "source_caption": row.source_caption,
            "bm25_text": row.bm25_text,
        }
        for row in lexical_rows
    ]
    return lexical_records, lexical_payload


def build_comparison_manifest(
    query: str,
    embedding_row: dict[str, object],
    hybrid_row: dict[str, object],
    semantic_report_path: str,
    hybrid_report_path: str,
) -> dict[str, object]:
    embedding_ids = [str(item["image_id"]) for item in embedding_row["items"]]
    hybrid_ids = [str(item["image_id"]) for item in hybrid_row["items"]]
    hybrid_set = set(hybrid_ids)
    embedding_set = set(embedding_ids)
    return {
        "query": query,
        "semantic_report_path": semantic_report_path,
        "hybrid_report_path": hybrid_report_path,
        "embedding_image_ids": embedding_ids,
        "hybrid_image_ids": hybrid_ids,
        "intersection_image_ids": [image_id for image_id in hybrid_ids if image_id in embedding_set],
        "only_embedding_image_ids": [image_id for image_id in embedding_ids if image_id not in hybrid_set],
        "only_hybrid_image_ids": [image_id for image_id in hybrid_ids if image_id not in embedding_set],
    }


def _copy_items(items: list[dict[str, object]], dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _fallback_rank(value: object, default_rank: int) -> int:
        try:
            rank = int(value)
        except (TypeError, ValueError):
            return default_rank
        return rank if rank > 0 else default_rank

    for rank, item in enumerate(items, start=1):
        src = Path(str(item["image_path"]))
        item_rank = _fallback_rank(item.get("rank", rank), rank)
        dest_path = dest_dir / f"{item_rank:02d}_{src.name}"
        shutil.copy2(src, dest_path)


def write_query_comparison_dirs(
    base_dir: Path,
    embedding_row: dict[str, object],
    hybrid_row: dict[str, object],
    semantic_report_path: str,
    hybrid_report_path: str,
) -> str:
    query = str(embedding_row["query"])
    query_dir = base_dir / _sanitize_query_dir_name(query)
    if query_dir.exists():
        shutil.rmtree(query_dir)
    query_dir.mkdir(parents=True, exist_ok=True)

    embedding_items = list(embedding_row["items"])
    hybrid_items = list(hybrid_row["items"])
    hybrid_ids = {str(item["image_id"]) for item in hybrid_items}
    embedding_ids = {str(item["image_id"]) for item in embedding_items}

    _copy_items(embedding_items, query_dir / "embedding")
    _copy_items(hybrid_items, query_dir / "hybrid")
    _copy_items(
        [item for item in embedding_items if str(item["image_id"]) not in hybrid_ids],
        query_dir / "only_embedding",
    )
    _copy_items(
        [item for item in hybrid_items if str(item["image_id"]) not in embedding_ids],
        query_dir / "only_hybrid",
    )

    manifest = build_comparison_manifest(
        query=query,
        embedding_row=embedding_row,
        hybrid_row=hybrid_row,
        semantic_report_path=semantic_report_path,
        hybrid_report_path=hybrid_report_path,
    )
    (query_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(query_dir.resolve())


def main() -> None:
    args = parse_args()
    semantic_report_path = Path(args.semantic_report)
    semantic_data = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    semantic_rows = semantic_data["query_results"]

    caption_records = load_caption_records(args.caption_result_json)
    lexical_records, lexical_payload = _lexical_caption_records(caption_records)
    lexical_backend = BM25RetrievalBackend(records=lexical_records)

    bm25_view_path = Path(args.bm25_view_json)
    bm25_view_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_view_path.write_text(
        json.dumps({"rows": lexical_payload, }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    hybrid_rows: list[dict[str, object]] = []
    for semantic_row in semantic_rows:
        semantic_items = _semantic_items_from_row(semantic_row)
        lexical_items = lexical_backend.search(
            query=str(semantic_row["search_query"]),
            top_k=max(args.top_k, args.lexical_top_n),
        )
        merged_items = rerank_semantic_tail_with_lexical_support(
            semantic_items=semantic_items,
            lexical_items=lexical_items,
            freeze_top_n=args.freeze_top_n,
            final_top_k=args.top_k,
            lexical_top_n=args.lexical_top_n,
        )
        hybrid_rows.append(
            {
                "query": semantic_row["query"],
                "search_query": semantic_row["search_query"],
                "expanded_english_queries": semantic_row.get("expanded_english_queries", []),
                "items": _format_items(merged_items),
            }
        )

    payload = _to_json_payload(
        caption_result_json=args.caption_result_json,
        backend="hybrid_embedding_bm25_main_experiment",
        top_k=args.top_k,
        query_results=hybrid_rows,
    )
    payload["summary"].update(
        {
            "semantic_source_report": str(semantic_report_path),
            "lexical_backend": "bm25",
            "bm25_view_json": str(bm25_view_path),
            "freeze_top_n": args.freeze_top_n,
            "lexical_top_n": args.lexical_top_n,
            "candidate_source": "semantic_top_k_only",
        }
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_recall_summary_csv(Path(args.output_csv), payload["query_results"])
    created_dirs = _organize_images_by_query(Path(args.hybrid_output_dir), payload["query_results"])
    payload["summary"]["retrieval_output_dir"] = str(Path(args.hybrid_output_dir).resolve())
    payload["summary"]["organized_query_dirs"] = created_dirs

    comparison_dirs: list[str] = []
    for embedding_row, hybrid_row in zip(semantic_rows, payload["query_results"], strict=True):
        comparison_dirs.append(
            write_query_comparison_dirs(
                base_dir=Path(args.comparison_dir),
                embedding_row=embedding_row,
                hybrid_row=hybrid_row,
                semantic_report_path=str(semantic_report_path),
                hybrid_report_path=str(out_json),
            )
        )
    payload["summary"]["comparison_dir"] = str(Path(args.comparison_dir).resolve())
    payload["summary"]["comparison_query_dirs"] = comparison_dirs
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
