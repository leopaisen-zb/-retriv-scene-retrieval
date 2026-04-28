"""文本检索图片实验入口。"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
from pathlib import Path

from pipelines.retrieval_pipeline import (
    BM25RetrievalBackend,
    HybridRetrievalBackend,
    RetrievalItem,
    RetrievalPipeline,
    Qwen3VLEmbeddingFaissBackend,
    TfidfFaissRetrievalBackend,
    load_caption_records,
    merge_multi_query_retrieval,
)
from utils.query_expansion import EnglishQueryExpander


LOGGER = logging.getLogger(__name__)
DEFAULT_SCENE_QUERIES: list[str] = [
    "vehicle ahead under strong backlighting in low-light conditions",
    "oncoming vehicle approaching in the opposite lane",
    "rainy road scene with the brake lights of the vehicle ahead illuminated",
]


def _contains_cjk(text: str) -> bool:
    """判断文本是否包含中日韩字符。"""
    return bool(re.search(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 参数对象。
    """
    parser = argparse.ArgumentParser(description="文本检索图片实验入口（向量召回）")
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="单条检索文本。与 --query-file 二选一；都不提供时使用默认场景 query",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default="",
        help="查询列表 JSON 文件路径，格式：{\"queries\": [\"q1\", \"q2\"]}",
    )
    parser.add_argument(
        "--caption-result-json",
        type=str,
        required=True,
        help="图片描述结果 JSON 路径（如 caption_results_all_qwen35.json）",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="qwen3vl_embedding_faiss",
        choices=["qwen3vl_embedding_faiss", "tfidf_faiss", "bm25", "hybrid_embedding_bm25"],
        help="检索后端类型",
    )
    parser.add_argument(
        "--embedder-script-path",
        type=str,
        default="/home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py",
        help="Qwen3-VL-Embedding 官方 qwen3_vl_embedding.py 路径",
    )
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default="/home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B",
        help="Qwen3-VL-Embedding 模型路径",
    )
    parser.add_argument(
        "--embedding-instruction",
        type=str,
        default="Represent the user's input.",
        help="embedding 指令文本",
    )
    parser.add_argument(
        "--embedding-torch-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="embedding 模型 dtype",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="embedding 编码批大小，减小可降低显存峰值",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="每个 query 最多返回条数（不超过语料总张数；语料很少时会条数不足）",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="仅保留相似度大于等于该阈值的召回；设置后先在全语料上打分再过滤并截断到 top_k",
    )
    parser.add_argument("--output-json", type=str, default="", help="JSON 输出路径")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="每 query 召回明细 CSV；留空且指定了 --output-json 时自动生成同名 _recall_summary.csv",
    )
    parser.add_argument(
        "--retrieval-output-dir",
        type=str,
        default="output/retrieval_output",
        help="按 query 分目录复制召回图片的根目录（相对当前工作目录或绝对路径）",
    )
    parser.add_argument(
        "--no-organize-by-query",
        action="store_true",
        help="关闭按 query 分目录复制图片",
    )
    parser.add_argument(
        "--no-expand-english-queries",
        action="store_true",
        help="关闭「中译英 + 多路相近英文 query 合并检索」；关闭后仅使用原始 query 单条检索（建议为英文）",
    )
    parser.add_argument(
        "--query-expand-model-path",
        type=str,
        default="/defaultShare/qwen-vl/models_cache_qwen3.5/Qwen/Qwen3.5-4B",
        help="用于生成英文扩展 query 的文本模型路径（与 embedding 模型可不同）",
    )
    parser.add_argument(
        "--expand-query-count",
        type=int,
        default=4,
        help="每条用户 query 生成的英文检索短语总数（1 条翻译 + 其余为相近场景）",
    )
    parser.add_argument(
        "--query-expand-max-new-tokens",
        type=int,
        default=512,
        help="扩展 query 生成时的 max_new_tokens",
    )
    return parser.parse_args()


def _parse_queries(query: str, query_file: str) -> list[dict[str, str]]:
    """解析查询文本列表（展示 query 与检索 query）。

    Args:
        query: 单条查询。
        query_file: 查询文件路径。

    Returns:
        list[dict[str, str]]: 每项包含 ``display_query`` 与 ``search_query``。

    Raises:
        ValueError: 输入不合法。
    """
    if query.strip() and query_file:
        raise ValueError("不能同时提供 --query 和 --query-file")

    if query_file:
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
                if not display_query:
                    display_query = search_query
                if not search_query:
                    search_query = display_query
                parsed.append({"display_query": display_query, "search_query": search_query})
                continue
            text = str(item).strip()
            if text:
                parsed.append({"display_query": text, "search_query": text})
        if not parsed:
            raise ValueError("query 文件中未找到有效查询语句")
        return parsed

    if query.strip():
        one = query.strip()
        return [{"display_query": one, "search_query": one}]
    return [{"display_query": q, "search_query": q} for q in DEFAULT_SCENE_QUERIES]


def _enrich_query_result_row(row: dict[str, object]) -> dict[str, object]:
    """为单条 query 结果补充便于统计的扁平字段。

    Args:
        row: 含 query 与 items 的字典。

    Returns:
        dict[str, object]: 补充 recalled_* 字段后的字典。
    """
    items = row.get("items", [])
    if not isinstance(items, list):
        return row
    paths: list[str] = []
    filenames: list[str] = []
    ids: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        p = str(it.get("image_path", "")).strip()
        if p:
            paths.append(p)
            filenames.append(Path(p).name)
        iid = str(it.get("image_id", "")).strip()
        if iid:
            ids.append(iid)
    row = dict(row)
    row["recalled_image_paths"] = paths
    row["recalled_image_filenames"] = filenames
    row["recalled_image_ids"] = ids
    row["recall_count"] = len(paths)
    return row


def _build_query_recall_summary(query_results: list[dict[str, object]]) -> list[dict[str, object]]:
    """构建顶层「每 query 召回哪些图」摘要列表。

    Args:
        query_results: 已 enrich 的 query 结果列表。

    Returns:
        list[dict[str, object]]: 每条含 query 与召回文件名/路径列表。
    """
    out: list[dict[str, object]] = []
    for row in query_results:
        if not isinstance(row, dict):
            continue
        q = str(row.get("query", ""))
        entry: dict[str, object] = {
            "query": q,
            "recall_count": row.get("recall_count", 0),
            "recalled_image_filenames": row.get("recalled_image_filenames", []),
            "recalled_image_paths": row.get("recalled_image_paths", []),
            "recalled_image_ids": row.get("recalled_image_ids", []),
        }
        exp = row.get("expanded_english_queries")
        if isinstance(exp, list):
            entry["expanded_english_queries"] = exp
        out.append(entry)
    return out


def _sanitize_query_dir_name(query: str) -> str:
    """将 query 转为可用作目录名的字符串（保留中文与常见标点，去掉路径非法字符）。

    Args:
        query: 原始 query 文本。

    Returns:
        str: 目录名。
    """
    text = query.strip()
    if not text:
        return "empty_query"
    for ch in ("\x00", "/", "\\"):
        text = text.replace(ch, "_")
    text = re.sub(r"[\r\n\t]+", "_", text)
    text = text.rstrip(". ")
    return text or "empty_query"


def _organize_images_by_query(
    base_dir: Path,
    query_results: list[dict[str, object]],
) -> list[str]:
    """按 query 在 base_dir/<query>/ 下复制召回图片，并写入 manifest.json。

    Args:
        base_dir: 根目录，例如 output/retrieval_output。
        query_results: 含 query 与 items 的结果列表。

    Returns:
        list[str]: 每个 query 对应的实际目录绝对路径列表。

    Raises:
        RuntimeError: 复制失败。
    """
    base_dir = base_dir.resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    created_dirs: list[str] = []

    for row in query_results:
        if not isinstance(row, dict):
            continue
        query = str(row.get("query", ""))
        folder_name = _sanitize_query_dir_name(query)
        dest_dir = base_dir / folder_name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, object]] = []
        items = row.get("items", [])
        if not isinstance(items, list):
            items = []

        for it in items:
            if not isinstance(it, dict):
                continue
            src = str(it.get("image_path", "")).strip()
            if not src:
                continue
            src_path = Path(src)
            if not src_path.is_file():
                LOGGER.warning("Source image missing, skip: %s", src)
                continue
            rank = int(it.get("rank", 0) or 0)
            score = it.get("score", "")
            orig_name = src_path.name
            dest_name = f"{rank:02d}_{orig_name}"
            dest_path = dest_dir / dest_name
            try:
                shutil.copy2(src_path, dest_path)
            except Exception as exc:  # pylint: disable=broad-except
                raise RuntimeError(f"复制图片失败：{src} -> {dest_path}，原因：{exc}") from exc
            manifest_rows.append(
                {
                    "rank": rank,
                    "score": score,
                    "source_path": str(src_path.resolve()),
                    "saved_as": dest_name,
                }
            )

        manifest_path = dest_dir / "manifest.json"
        manifest_obj: dict[str, object] = {
            "query": query,
            "folder_name": folder_name,
            "items": manifest_rows,
        }
        exp = row.get("expanded_english_queries")
        if isinstance(exp, list):
            manifest_obj["expanded_english_queries"] = exp
        manifest_path.write_text(
            json.dumps(manifest_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        created_dirs.append(str(dest_dir.resolve()))
        LOGGER.info("Organized %d images under %s", len(manifest_rows), dest_dir)

    return created_dirs


def _write_recall_summary_csv(
    path: Path,
    query_results: list[dict[str, object]],
) -> None:
    """写入扁平 CSV：每行一条召回，便于透视表统计。

    Args:
        path: CSV 输出路径。
        query_results: 含 query 与 items 的结果列表。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query",
        "expanded_english_queries",
        "rank",
        "image_filename",
        "image_path",
        "image_id",
        "score",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in query_results:
            if not isinstance(row, dict):
                continue
            q = str(row.get("query", ""))
            exp_raw = row.get("expanded_english_queries", [])
            if isinstance(exp_raw, list):
                exp_col = " | ".join(str(x) for x in exp_raw)
            else:
                exp_col = ""
            items = row.get("items", [])
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                ip = str(it.get("image_path", "")).strip()
                writer.writerow(
                    {
                        "query": q,
                        "expanded_english_queries": exp_col,
                        "rank": it.get("rank", ""),
                        "image_filename": Path(ip).name if ip else "",
                        "image_path": ip,
                        "image_id": str(it.get("image_id", "")),
                        "score": it.get("score", ""),
                    }
                )


def _to_json_payload(
    caption_result_json: str,
    backend: str,
    top_k: int,
    query_results: list[dict[str, object]],
) -> dict[str, object]:
    """将检索结果转换为 JSON 可序列化结构。"""
    enriched = [_enrich_query_result_row(dict(r)) for r in query_results]
    return {
        "summary": {
            "backend": backend,
            "caption_result_json": caption_result_json,
            "total_queries": len(enriched),
            "top_k": top_k,
            "note": "无 GT 评估，供人工判读。query_recall_summary 与每条 query 的 recalled_* 字段便于统计。",
        },
        "query_recall_summary": _build_query_recall_summary(enriched),
        "query_results": enriched,
    }


def _format_items(items: list[RetrievalItem]) -> list[dict[str, object]]:
    """格式化检索结果项。"""
    return [
        {
            "rank": rank + 1,
            "image_id": item.image_id,
            "image_path": item.image_path,
            "score": item.score,
            "caption": item.caption,
            "source_result_file": item.source_result_file,
        }
        for rank, item in enumerate(items)
    ]


def main() -> None:
    """程序入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    queries = _parse_queries(query=args.query, query_file=args.query_file)
    records = load_caption_records(result_json_path=args.caption_result_json)
    if args.backend == "qwen3vl_embedding_faiss":
        backend = Qwen3VLEmbeddingFaissBackend(
            records=records,
            embedder_script_path=args.embedder_script_path,
            embedding_model_path=args.embedding_model_path,
            instruction=args.embedding_instruction,
            torch_dtype=args.embedding_torch_dtype,
            encode_batch_size=args.embedding_batch_size,
        )
    elif args.backend == "bm25":
        backend = BM25RetrievalBackend(records=records)
    elif args.backend == "hybrid_embedding_bm25":
        backend = HybridRetrievalBackend(
            semantic_backend=Qwen3VLEmbeddingFaissBackend(
                records=records,
                embedder_script_path=args.embedder_script_path,
                embedding_model_path=args.embedding_model_path,
                instruction=args.embedding_instruction,
                torch_dtype=args.embedding_torch_dtype,
                encode_batch_size=args.embedding_batch_size,
            ),
            lexical_backend=BM25RetrievalBackend(records=records),
        )
    else:
        backend = TfidfFaissRetrievalBackend(records=records)
    pipeline = RetrievalPipeline(backend=backend)

    corpus_n = len(records)
    expander: EnglishQueryExpander | None = None
    if not args.no_expand_english_queries:
        try:
            expander = EnglishQueryExpander(
                model_path=args.query_expand_model_path,
                max_new_tokens=args.query_expand_max_new_tokens,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"英文扩展 query 模型初始化失败：{exc}") from exc

    query_results: list[dict[str, object]] = []
    for q in queries:
        display_query = q["display_query"].strip()
        search_query = q["search_query"].strip()
        LOGGER.info("Running retrieval for user query: %s", display_query)
        if expander is not None:
            english_queries = expander.expand(
                user_query=display_query,
                total_phrases=max(2, args.expand_query_count),
            )
            LOGGER.info("Expanded English queries: %s", english_queries)
            merge_k = max(1, corpus_n)
            items = merge_multi_query_retrieval(
                backend=backend,
                queries=english_queries,
                top_k=merge_k,
                corpus_size=corpus_n,
            )
            if args.min_score is not None:
                items = [it for it in items if it.score >= args.min_score]
            items = items[: args.top_k]
            query_results.append(
                {
                    "query": display_query,
                    "search_query": search_query,
                    "expanded_english_queries": english_queries,
                    "items": _format_items(items=items),
                }
            )
        else:
            if _contains_cjk(search_query):
                raise ValueError(
                    "关闭扩展模式下要求检索输入为英文。"
                    f"请在 query 文件为该条提供 english_query，当前 search_query={search_query!r}"
                )
            if args.min_score is not None:
                search_k = max(1, corpus_n)
            else:
                search_k = min(args.top_k, max(1, corpus_n))
            items = pipeline.search(query=search_query, top_k=search_k)
            if args.min_score is not None:
                items = [it for it in items if it.score >= args.min_score]
            items = items[: args.top_k]
            query_results.append(
                {
                    "query": display_query,
                    "search_query": search_query,
                    "expanded_english_queries": [search_query],
                    "items": _format_items(items=items),
                }
            )

    payload = _to_json_payload(
        caption_result_json=args.caption_result_json,
        backend=args.backend,
        top_k=args.top_k,
        query_results=query_results,
    )
    summary_block = payload.get("summary")
    if isinstance(summary_block, dict):
        summary_block["corpus_image_count"] = corpus_n
        summary_block["min_score"] = args.min_score
        summary_block["expand_english_queries"] = not args.no_expand_english_queries
        summary_block["expand_query_count"] = args.expand_query_count
        summary_block["query_expand_model_path"] = (
            args.query_expand_model_path if not args.no_expand_english_queries else None
        )
        summary_block["embedder_script_path"] = (
            args.embedder_script_path if args.backend == "qwen3vl_embedding_faiss" else None
        )
        summary_block["embedding_model_path"] = (
            args.embedding_model_path if args.backend == "qwen3vl_embedding_faiss" else None
        )
        summary_block["embedding_instruction"] = (
            args.embedding_instruction if args.backend == "qwen3vl_embedding_faiss" else None
        )
        summary_block["retrieval_note"] = (
            f"当前语料共 {corpus_n} 张；每个 query 最多返回 min(top_k, 语料张数) 条。"
            " 若 top_k 大于等于语料张数，则每个 query 都会列出全部语料。"
            " 需要更少结果时可调小 --top-k，或使用 --min-score 做相似度截断。"
            " 默认启用英文扩展：每条用户 query 会生成多条英文相近检索语，分别 embedding 检索后在图片维度取最大相似度再取 top_k。"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    out_path: Path | None = None
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        csv_path = Path(args.output_csv) if args.output_csv else out_path.with_name(
            f"{out_path.stem}_recall_summary.csv"
        )
        _write_recall_summary_csv(csv_path, payload["query_results"])  # type: ignore[arg-type]
        LOGGER.info("JSON saved to %s", out_path)
        LOGGER.info("Recall summary CSV saved to %s", csv_path)

    if not args.no_organize_by_query:
        base = Path(args.retrieval_output_dir)
        if not base.is_absolute():
            base = Path.cwd() / base
        try:
            dirs = _organize_images_by_query(
                base_dir=base,
                query_results=payload["query_results"],  # type: ignore[arg-type]
            )
            summary_block = payload.get("summary")
            if isinstance(summary_block, dict):
                summary_block["retrieval_output_dir"] = str(base.resolve())
                summary_block["organized_query_dirs"] = dirs
            if out_path is not None:
                out_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"按 query 分目录复制图片失败：{exc}") from exc


if __name__ == "__main__":
    main()
