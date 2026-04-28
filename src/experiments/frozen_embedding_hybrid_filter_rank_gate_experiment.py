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
from pipelines.hybrid_rank_gate_filter import filter_semantic_candidates_by_rank_gates
from pipelines.retrieval_pipeline import (
    BM25RetrievalBackend,
    CaptionRecord,
    RetrievalItem,
    load_caption_records,
)
from utils.bm25_lexical_view import build_bm25_view_rows

_FIXED_CANDIDATE_TOP_K = 20
_FIXED_SEMANTIC_TOP_N = 10
_FIXED_LEXICAL_TOP_N = 8
_FIXED_OUTPUT_MAX_K = 10
_FIXED_BM25_TOP_N = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结 embedding 基线的 rank-gate filtering 实验")
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
        default="output/artifacts/bm25_views/hybrid_filter_rank_gate.json",
    )
    parser.add_argument(
        "--output-json",
        default="output/reports/retrieval_issues_all_hybrid_filter_rank_gate.json",
    )
    parser.add_argument(
        "--output-csv",
        default="output/reports/retrieval_issues_all_hybrid_filter_rank_gate_recall_summary.csv",
    )
    parser.add_argument(
        "--filtered-output-dir",
        default="output/retrieval_output_hybrid_filter_rank_gate",
    )
    parser.add_argument(
        "--comparison-dir",
        default="output/comparisons/hybrid_filter_rank_gate",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=_FIXED_CANDIDATE_TOP_K,
        help=(
            f"Fixed contract value: {_FIXED_CANDIDATE_TOP_K}. "
            "Override will raise at runtime."
        ),
    )
    parser.add_argument(
        "--semantic-top-n",
        type=int,
        default=_FIXED_SEMANTIC_TOP_N,
        help=(
            f"Fixed contract value: {_FIXED_SEMANTIC_TOP_N}. "
            "Override will raise at runtime."
        ),
    )
    parser.add_argument(
        "--lexical-top-n",
        type=int,
        default=_FIXED_LEXICAL_TOP_N,
        help=(
            f"Fixed contract value: {_FIXED_LEXICAL_TOP_N}. "
            "Override will raise at runtime."
        ),
    )
    parser.add_argument(
        "--output-max-k",
        type=int,
        default=_FIXED_OUTPUT_MAX_K,
        help=(
            f"Fixed contract value: {_FIXED_OUTPUT_MAX_K}. "
            "Override will raise at runtime."
        ),
    )
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


def _validate_fixed_rank_gate_contract(args: argparse.Namespace) -> None:
    if args.candidate_top_k != _FIXED_CANDIDATE_TOP_K:
        raise ValueError(f"candidate_top_k 必须固定为 {_FIXED_CANDIDATE_TOP_K}")
    if args.semantic_top_n != _FIXED_SEMANTIC_TOP_N:
        raise ValueError(f"semantic_top_n 必须固定为 {_FIXED_SEMANTIC_TOP_N}")
    if args.lexical_top_n != _FIXED_LEXICAL_TOP_N:
        raise ValueError(f"lexical_top_n 必须固定为 {_FIXED_LEXICAL_TOP_N}")
    if args.output_max_k != _FIXED_OUTPUT_MAX_K:
        raise ValueError(f"output_max_k 必须固定为 {_FIXED_OUTPUT_MAX_K}")


def build_filter_manifest(
    query: str,
    embedding_row: dict[str, object],
    bm25_row: dict[str, object],
    filtered_row: dict[str, object],
    candidate_top_k: int,
    semantic_top_n: int,
    semantic_report_path: str,
    filtered_report_path: str,
) -> dict[str, object]:
    embedding_top20_items = list(embedding_row["items"])[:candidate_top_k]
    embedding_top10_items = embedding_top20_items[:semantic_top_n]
    bm25_items = list(bm25_row["items"])[:_FIXED_BM25_TOP_N]
    filtered_items = list(filtered_row["items"])

    final_ids = [str(item["image_id"]) for item in filtered_items]
    final_set = set(final_ids)
    embedding_top20_ids = [str(item["image_id"]) for item in embedding_top20_items]
    embedding_top10_ids = [str(item["image_id"]) for item in embedding_top10_items]
    bm25_ids = [str(item["image_id"]) for item in bm25_items]

    return {
        "query": query,
        "semantic_report_path": semantic_report_path,
        "filtered_report_path": filtered_report_path,
        "embedding_top20_image_ids": embedding_top20_ids,
        "embedding_top10_image_ids": embedding_top10_ids,
        "bm25_top8_image_ids": bm25_ids,
        "final_filtered_image_ids": final_ids,
        "filtered_out_image_ids": [
            image_id for image_id in embedding_top10_ids if image_id not in final_set
        ],
        "final_result_count": len(final_ids),
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
        shutil.copy2(src, dest_dir / f"{item_rank:02d}_{src.name}")


def write_query_filter_comparison_dirs(
    base_dir: Path,
    embedding_row: dict[str, object],
    bm25_row: dict[str, object],
    filtered_row: dict[str, object],
    candidate_top_k: int,
    semantic_top_n: int,
    semantic_report_path: str,
    filtered_report_path: str,
) -> str:
    query = str(embedding_row["query"])
    query_dir = base_dir / _sanitize_query_dir_name(query)
    if query_dir.exists():
        shutil.rmtree(query_dir)
    query_dir.mkdir(parents=True, exist_ok=True)

    embedding_top20_items = list(embedding_row["items"])[:candidate_top_k]
    embedding_top10_items = embedding_top20_items[:semantic_top_n]
    bm25_items = list(bm25_row["items"])[:_FIXED_BM25_TOP_N]
    filtered_items = list(filtered_row["items"])
    filtered_ids = {str(item["image_id"]) for item in filtered_items}

    _copy_items(embedding_top20_items, query_dir / "embedding_top20")
    _copy_items(embedding_top10_items, query_dir / "embedding_top10")
    _copy_items(bm25_items, query_dir / "bm25_top8_candidates")
    _copy_items(filtered_items, query_dir / "final_filtered")
    _copy_items(
        [
            item
            for item in embedding_top10_items
            if str(item["image_id"]) not in filtered_ids
        ],
        query_dir / "filtered_out_from_embedding_top10",
    )

    manifest = build_filter_manifest(
        query=query,
        embedding_row=embedding_row,
        bm25_row=bm25_row,
        filtered_row=filtered_row,
        candidate_top_k=candidate_top_k,
        semantic_top_n=semantic_top_n,
        semantic_report_path=semantic_report_path,
        filtered_report_path=filtered_report_path,
    )
    (query_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(query_dir.resolve())


def main() -> None:
    args = parse_args()
    _validate_fixed_rank_gate_contract(args)
    semantic_report_path = Path(args.semantic_report)
    semantic_data = json.loads(semantic_report_path.read_text(encoding="utf-8"))
    semantic_rows = semantic_data["query_results"]

    caption_records = load_caption_records(args.caption_result_json)
    lexical_records, lexical_payload = _lexical_caption_records(caption_records)
    lexical_backend = BM25RetrievalBackend(records=lexical_records)

    bm25_view_path = Path(args.bm25_view_json)
    bm25_view_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_view_path.write_text(
        json.dumps({"rows": lexical_payload}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    bm25_rows: list[dict[str, object]] = []
    filtered_rows: list[dict[str, object]] = []

    for semantic_row in semantic_rows:
        semantic_items = _semantic_items_from_row(semantic_row)
        lexical_items = lexical_backend.search(
            query=str(semantic_row["search_query"]),
            top_k=args.lexical_top_n,
        )
        filtered_items = filter_semantic_candidates_by_rank_gates(
            semantic_items=semantic_items,
            lexical_items=lexical_items,
            candidate_top_k=args.candidate_top_k,
            semantic_top_n=args.semantic_top_n,
            lexical_top_n=args.lexical_top_n,
            output_max_k=args.output_max_k,
        )

        bm25_rows.append(
            {
                "query": semantic_row["query"],
                "search_query": semantic_row["search_query"],
                "expanded_english_queries": semantic_row.get("expanded_english_queries", []),
                "items": _format_items(lexical_items[: args.lexical_top_n]),
            }
        )
        filtered_rows.append(
            {
                "query": semantic_row["query"],
                "search_query": semantic_row["search_query"],
                "expanded_english_queries": semantic_row.get("expanded_english_queries", []),
                "items": _format_items(filtered_items),
            }
        )

    payload = _to_json_payload(
        caption_result_json=args.caption_result_json,
        backend="hybrid_embedding_bm25_filter_rank_gate",
        top_k=args.output_max_k,
        query_results=filtered_rows,
    )
    payload["summary"].update(
        {
            "semantic_source_report": str(semantic_report_path),
            "lexical_backend": "bm25",
            "bm25_view_json": str(bm25_view_path),
            "candidate_top_k": args.candidate_top_k,
            "semantic_top_n": args.semantic_top_n,
            "lexical_top_n": args.lexical_top_n,
            "output_max_k": args.output_max_k,
            "candidate_source": "semantic_top_k_only",
            "filter_strategy": "semantic_rank_gate_intersect_bm25_rank_gate",
        }
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_recall_summary_csv(Path(args.output_csv), payload["query_results"])

    created_dirs = _organize_images_by_query(Path(args.filtered_output_dir), payload["query_results"])
    payload["summary"]["retrieval_output_dir"] = str(Path(args.filtered_output_dir).resolve())
    payload["summary"]["organized_query_dirs"] = created_dirs

    comparison_dirs: list[str] = []
    for embedding_row, bm25_row, filtered_row in zip(
        semantic_rows,
        bm25_rows,
        payload["query_results"],
        strict=True,
    ):
        comparison_dirs.append(
            write_query_filter_comparison_dirs(
                base_dir=Path(args.comparison_dir),
                embedding_row=embedding_row,
                bm25_row=bm25_row,
                filtered_row=filtered_row,
                candidate_top_k=args.candidate_top_k,
                semantic_top_n=args.semantic_top_n,
                semantic_report_path=str(semantic_report_path),
                filtered_report_path=str(out_json),
            )
        )

    payload["summary"]["comparison_dir"] = str(Path(args.comparison_dir).resolve())
    payload["summary"]["comparison_query_dirs"] = comparison_dirs
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
