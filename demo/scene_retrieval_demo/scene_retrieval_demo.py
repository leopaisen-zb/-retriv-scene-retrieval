"""Self-contained demo for caption-based scene retrieval.

Commands:
  caption   Generate image captions with Qwen3.5-VL.
  retrieve  Search caption text with BM25 or Qwen3-VL-Embedding + FAISS.
  eval      Strict hash-deduplicated Top-k recall evaluation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_QUERY_LABEL_MAP = {
    "vehicle ahead under strong backlighting in low-light conditions": "低光照下，前车强逆光",
    "oncoming vehicle approaching in the opposite lane": "对向来车",
    "rainy road scene with the brake lights of the vehicle ahead illuminated": "雨天，前车刹车灯亮起",
}


@dataclass(frozen=True)
class CaptionRecord:
    image_id: str
    image_path: str
    caption: str
    source_result_file: str


@dataclass(frozen=True)
class RetrievalItem:
    image_id: str
    image_path: str
    score: float
    caption: str
    source_result_file: str


@dataclass(frozen=True)
class DedupedImageMetadata:
    image_hash: str
    representative_path: str
    duplicate_paths: tuple[str, ...]
    labels: set[str]

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_paths)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_top_ks(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
        values = [int(chunk) for chunk in chunks]
    else:
        values = [int(item) for item in value]
    if not values or any(item <= 0 for item in values):
        raise ValueError("top_k values must be positive integers")
    return values


def parse_backends(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        backends = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    else:
        backends = [str(item).strip() for item in value if str(item).strip()]
    allowed = {"bm25", "qwen3vl_embedding_faiss"}
    invalid = sorted(set(backends) - allowed)
    if invalid:
        raise ValueError(f"unsupported backend(s): {invalid}")
    if not backends:
        raise ValueError("at least one backend is required")
    return backends


def collect_images(image_root: str | Path, max_images: int = 0) -> list[Path]:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"image root does not exist: {root}")
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if max_images > 0:
        images = images[:max_images]
    return images


def tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", text.lower())


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"image file does not exist: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_path(value: str | Path) -> str:
    return str(Path(value).resolve())


def load_caption_records(result_json_path: str | Path) -> list[CaptionRecord]:
    path = Path(result_json_path)
    raw = load_json(path)
    rows = raw.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("caption JSON must contain a results list")

    records = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("success", False):
            continue
        image_path = str(row.get("image_path", "")).strip()
        caption = str(row.get("caption", "")).strip()
        if not image_path or not caption:
            continue
        records.append(
            CaptionRecord(
                image_id=Path(image_path).stem,
                image_path=image_path,
                caption=caption,
                source_result_file=str(path),
            )
        )
    return records


def parse_query_file(query_file: str | Path) -> list[dict[str, str]]:
    raw = load_json(query_file)
    rows = raw.get("queries", [])
    if not isinstance(rows, list):
        raise ValueError("query file must contain a queries list")
    queries = []
    for row in rows:
        if isinstance(row, str):
            query = row.strip()
            if query:
                queries.append({"display_query": query, "search_query": query})
            continue
        if not isinstance(row, dict):
            continue
        display_query = str(row.get("query") or row.get("display_query") or "").strip()
        search_query = str(row.get("english_query") or row.get("search_query") or "").strip()
        if display_query or search_query:
            queries.append(
                {
                    "display_query": display_query or search_query,
                    "search_query": search_query or display_query,
                }
            )
    if not queries:
        raise ValueError("no valid query found")
    return queries


def label_for_path(image_path: str, label_root: str | Path) -> str | None:
    try:
        rel = Path(image_path).resolve().relative_to(Path(label_root).resolve())
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def build_deduplicated_corpus(
    records: list[CaptionRecord],
    label_root: str | Path,
) -> tuple[list[CaptionRecord], dict[str, DedupedImageMetadata]]:
    selected: dict[str, CaptionRecord] = {}
    duplicate_paths: dict[str, list[str]] = {}
    labels: dict[str, set[str]] = {}

    for record in records:
        image_path = resolved_path(record.image_path)
        image_hash = sha256_file(image_path)
        duplicate_paths.setdefault(image_hash, []).append(image_path)
        label = label_for_path(image_path, label_root)
        labels.setdefault(image_hash, set())
        if label:
            labels[image_hash].add(label)
        if image_hash not in selected:
            selected[image_hash] = CaptionRecord(
                image_id=image_hash,
                image_path=image_path,
                caption=record.caption,
                source_result_file=record.source_result_file,
            )

    corpus = []
    index = {}
    for image_hash, record in selected.items():
        representative_path = resolved_path(record.image_path)
        corpus.append(
            CaptionRecord(
                image_id=image_hash,
                image_path=representative_path,
                caption=record.caption,
                source_result_file=record.source_result_file,
            )
        )
        index[image_hash] = DedupedImageMetadata(
            image_hash=image_hash,
            representative_path=representative_path,
            duplicate_paths=tuple(duplicate_paths[image_hash]),
            labels=labels[image_hash],
        )
    return corpus, index


def image_hash_by_path(image_index: dict[str, DedupedImageMetadata]) -> dict[str, str]:
    lookup = {}
    for image_hash, meta in image_index.items():
        lookup[meta.representative_path] = image_hash
        for path in meta.duplicate_paths:
            lookup[path] = image_hash
    return lookup


def hash_lookup(image_hash_by_path_map: dict[str, str], image_path: str) -> str:
    found = image_hash_by_path_map.get(image_path)
    if found is not None:
        return found
    found = image_hash_by_path_map.get(resolved_path(image_path))
    if found is None:
        raise ValueError(f"retrieval item path is not in hash lookup: {image_path}")
    return found


def evaluate_fixed_topk_hash_recall(
    items: list[RetrievalItem],
    gt_hashes: set[str],
    top_ks: list[int],
    image_hash_by_path: dict[str, str],
) -> list[dict[str, Any]]:
    if not gt_hashes:
        raise ValueError("gt_hashes cannot be empty")
    rows = []
    for top_k in top_ks:
        candidate_items = items[:top_k]
        candidate_hashes = {
            hash_lookup(image_hash_by_path, item.image_path)
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


class CaptionPipeline:
    def __init__(self, model_path: str, prompt_config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.prompt_config = prompt_config
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def infer_one(self, image_path: str, max_new_tokens: int) -> str:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path).convert("RGB")
        messages = self.build_messages(image)
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids, strict=True)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def build_messages(self, image: Any) -> list[dict[str, Any]]:
        messages = [
            {
                "role": "system",
                "content": self.prompt_config["system_instruction"],
            }
        ]
        messages.extend(self.prompt_config.get("few_shots", []))
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": self.prompt_config["default_user_query"],
                    },
                ],
            }
        )
        return messages


class BM25Backend:
    def __init__(self, records: list[CaptionRecord]) -> None:
        from rank_bm25 import BM25Okapi

        if not records:
            raise ValueError("records cannot be empty")
        self.records = records
        self.tokenized_corpus = [tokenize_bm25(record.caption) for record in records]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        scores = self.bm25.get_scores(tokenize_bm25(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        items = []
        for idx, score in ranked[: min(top_k, len(self.records))]:
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


class Qwen3VLEmbeddingFaissBackend:
    def __init__(
        self,
        records: list[CaptionRecord],
        embedder_script_path: str,
        embedding_model_path: str,
        instruction: str,
        torch_dtype: str,
        batch_size: int,
    ) -> None:
        import faiss
        import numpy as np
        import torch

        if not records:
            raise ValueError("records cannot be empty")
        self.faiss = faiss
        self.np = np
        self.records = records
        self.instruction = instruction
        self.batch_size = max(1, int(batch_size))
        self.embedder = self.load_embedder(embedder_script_path, embedding_model_path, torch_dtype, torch)
        vectors = self.encode_texts([record.caption for record in records])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @staticmethod
    def load_embedder(script_path: str, model_path: str, torch_dtype: str, torch_module: Any) -> Any:
        script = Path(script_path)
        if not script.is_file():
            raise FileNotFoundError(f"embedder script not found: {script}")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"embedding model not found: {model_path}")
        spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_demo", str(script))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load embedder script: {script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        dtype = {
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "float32": torch_module.float32,
        }.get(torch_dtype, torch_module.bfloat16)
        return module.Qwen3VLEmbedder(model_name_or_path=model_path, torch_dtype=dtype)

    def encode_texts(self, texts: list[str]) -> Any:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            part = texts[start:start + self.batch_size]
            model_inputs = [
                {"text": text, "instruction": self.instruction}
                for text in part
            ]
            embeddings = self.embedder.process(model_inputs, normalize=True)
            if hasattr(embeddings, "detach"):
                embeddings = embeddings.float().detach().cpu().numpy()
            chunks.append(self.np.asarray(embeddings, dtype=self.np.float32))
        return self.np.concatenate(chunks, axis=0)

    def search(self, query: str, top_k: int) -> list[RetrievalItem]:
        query_vector = self.encode_texts([query])
        scores, indices = self.index.search(query_vector, min(top_k, len(self.records)))
        items = []
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


def build_backend(name: str, records: list[CaptionRecord], config: dict[str, Any]) -> Any:
    if name == "bm25":
        return BM25Backend(records)
    if name == "qwen3vl_embedding_faiss":
        embedding = config["embedding"]
        return Qwen3VLEmbeddingFaissBackend(
            records=records,
            embedder_script_path=embedding["embedder_script_path"],
            embedding_model_path=embedding["embedding_model_path"],
            instruction=embedding.get("instruction", "Represent the user's input."),
            torch_dtype=embedding.get("torch_dtype", "bfloat16"),
            batch_size=int(embedding.get("batch_size", 8)),
        )
    raise ValueError(f"unsupported backend: {name}")


def ranked_item_payload(items: list[RetrievalItem]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "image_id": item.image_id,
            "image_path": item.image_path,
            "image_filename": Path(item.image_path).name,
            "score": item.score,
            "caption": item.caption,
        }
        for rank, item in enumerate(items, start=1)
    ]


def ranked_item_eval_payload(
    items: list[RetrievalItem],
    gt_hashes: set[str],
    hash_by_path: dict[str, str],
    image_index: dict[str, DedupedImageMetadata],
) -> list[dict[str, Any]]:
    rows = []
    for rank, item in enumerate(items, start=1):
        image_hash = hash_lookup(hash_by_path, item.image_path)
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


def write_eval_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
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


def command_caption(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    caption_config = config["caption"]
    prompt_config = load_json(caption_config["prompt_file"])
    images = collect_images(
        caption_config["image_root"],
        max_images=int(args.max_images or caption_config.get("max_images", 0)),
    )
    if not images:
        raise RuntimeError("no images found")

    pipeline = CaptionPipeline(caption_config["model_path"], prompt_config)
    output_json = Path(args.output_json or caption_config["output_json"])
    output_jsonl = Path(args.output_jsonl or caption_config.get("output_jsonl", ""))
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.write_text("", encoding="utf-8")

    results = []
    timings = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] caption {image_path}")
        started = time.perf_counter()
        try:
            caption = pipeline.infer_one(
                str(image_path),
                max_new_tokens=int(args.max_new_tokens or caption_config["max_new_tokens"]),
            )
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            row = {
                "image_path": str(image_path),
                "success": True,
                "inference_seconds": round(elapsed, 6),
                "caption": caption,
            }
        except Exception as exc:  # pylint: disable=broad-except
            row = {
                "image_path": str(image_path),
                "success": False,
                "error": str(exc),
            }
        results.append(row)
        if output_jsonl:
            with output_jsonl.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    avg = sum(timings) / len(timings) if timings else None
    write_json(
        output_json,
        {
            "summary": {
                "model_path": caption_config["model_path"],
                "image_root": caption_config["image_root"],
                "total_images": len(images),
                "success_images": len(timings),
                "failed_images": len(images) - len(timings),
                "average_inference_seconds": round(avg, 6) if avg else None,
                "max_new_tokens": int(args.max_new_tokens or caption_config["max_new_tokens"]),
            },
            "results": results,
        },
    )
    print(f"caption output: {output_json}")


def load_demo_records(config: dict[str, Any], deduplicate: bool) -> tuple[list[CaptionRecord], dict[str, DedupedImageMetadata] | None]:
    retrieval = config["retrieval"]
    records = load_caption_records(retrieval["caption_result_json"])
    if deduplicate:
        corpus, index = build_deduplicated_corpus(records, retrieval["label_root"])
        return corpus, index
    return records, None


def command_retrieve(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    retrieval = config["retrieval"]
    backends = parse_backends(args.backends or retrieval.get("backends", ["bm25"]))
    top_k = int(args.top_k or retrieval.get("top_k", 30))
    deduplicate = bool(retrieval.get("deduplicate_by_hash", True))
    records, image_index = load_demo_records(config, deduplicate=deduplicate)
    queries = parse_query_file(retrieval["query_file"])

    results = []
    for backend_name in backends:
        started = time.perf_counter()
        backend = build_backend(backend_name, records, config)
        build_seconds = time.perf_counter() - started
        query_rows = []
        for query_row in queries:
            query_started = time.perf_counter()
            items = backend.search(query_row["search_query"], top_k=top_k)
            query_seconds = time.perf_counter() - query_started
            query_rows.append(
                {
                    "query": query_row["display_query"],
                    "search_query": query_row["search_query"],
                    "query_seconds": round(query_seconds, 6),
                    "items": ranked_item_payload(items),
                }
            )
        results.append(
            {
                "backend": backend_name,
                "build_seconds": round(build_seconds, 6),
                "queries": query_rows,
            }
        )

    output_json = args.output_json or retrieval["output_json"]
    write_json(
        output_json,
        {
            "summary": {
                "caption_result_json": retrieval["caption_result_json"],
                "query_file": retrieval["query_file"],
                "deduplicated": deduplicate,
                "corpus_count": len(records),
                "unique_image_count": len(image_index) if image_index else None,
                "top_k": top_k,
                "backends": backends,
            },
            "results": results,
        },
    )
    print(f"retrieval output: {output_json}")


def command_eval(args: argparse.Namespace) -> None:
    config = load_json(args.config)
    retrieval = config["retrieval"]
    backends = parse_backends(args.backends or retrieval.get("backends", ["bm25"]))
    top_ks = parse_top_ks(args.top_ks or retrieval.get("top_ks", [10, 20, 30]))
    raw_records = load_caption_records(retrieval["caption_result_json"])
    corpus, index = build_deduplicated_corpus(raw_records, retrieval["label_root"])
    hash_by_path = image_hash_by_path(index)
    queries = parse_query_file(retrieval["query_file"])

    csv_rows = []
    query_results = []
    for query_row in queries:
        display_query = query_row["display_query"]
        label = DEFAULT_QUERY_LABEL_MAP.get(display_query)
        if label is None:
            raise ValueError(f"missing GT label mapping for query: {display_query}")
        gt_hashes = {
            image_hash
            for image_hash, meta in index.items()
            if label in meta.labels
        }
        backend_results = []
        for backend_name in backends:
            build_started = time.perf_counter()
            backend = build_backend(backend_name, corpus, config)
            build_seconds = time.perf_counter() - build_started
            query_started = time.perf_counter()
            items = backend.search(query_row["search_query"], top_k=len(corpus))
            query_seconds = time.perf_counter() - query_started
            recall_rows = evaluate_fixed_topk_hash_recall(
                items=items,
                gt_hashes=gt_hashes,
                top_ks=top_ks,
                image_hash_by_path=hash_by_path,
            )
            for row in recall_rows:
                csv_rows.append(
                    {
                        "query": display_query,
                        "backend": backend_name,
                        "top_k": row["top_k"],
                        "corpus_unique_count": len(corpus),
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
                    "ranked_items": ranked_item_eval_payload(items, gt_hashes, hash_by_path, index),
                }
            )
        query_results.append(
            {
                "query": display_query,
                "search_query": query_row["search_query"],
                "gt_label": label,
                "gt_count": len(gt_hashes),
                "gt_hashes": sorted(gt_hashes),
                "backends": backend_results,
            }
        )

    output_json = args.output_json or retrieval["eval_output_json"]
    output_csv = args.output_csv or retrieval["eval_output_csv"]
    write_json(
        output_json,
        {
            "summary": {
                "caption_result_json": retrieval["caption_result_json"],
                "query_file": retrieval["query_file"],
                "label_root": retrieval["label_root"],
                "raw_caption_record_count": len(raw_records),
                "deduplicated_corpus_count": len(corpus),
                "top_ks": top_ks,
                "backends": backends,
                "retrieval_scope": "all_deduplicated_images",
                "gt_usage": "GT hashes are used only for evaluation, not retrieval filtering.",
            },
            "deduplicated_images": [
                {
                    "image_hash": image_hash,
                    "representative_path": meta.representative_path,
                    "duplicate_paths": list(meta.duplicate_paths),
                    "duplicate_count": meta.duplicate_count,
                    "labels": sorted(meta.labels),
                }
                for image_hash, meta in index.items()
            ],
            "query_results": query_results,
        },
    )
    write_eval_csv(output_csv, csv_rows)
    print(f"eval json: {output_json}")
    print(f"eval csv: {output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caption-based scene retrieval demo")
    parser.add_argument(
        "--config",
        default="demo/scene_retrieval_demo/config/demo_config.json",
        help="Demo config JSON path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    caption = subparsers.add_parser("caption", help="Generate captions for images")
    caption.add_argument("--config", default=argparse.SUPPRESS)
    caption.add_argument("--max-images", type=int, default=0)
    caption.add_argument("--max-new-tokens", type=int, default=0)
    caption.add_argument("--output-json", default="")
    caption.add_argument("--output-jsonl", default="")
    caption.set_defaults(func=command_caption)

    retrieve = subparsers.add_parser("retrieve", help="Run BM25 or embedding retrieval")
    retrieve.add_argument("--config", default=argparse.SUPPRESS)
    retrieve.add_argument("--backends", default="", help="Comma-separated backends")
    retrieve.add_argument("--top-k", type=int, default=0)
    retrieve.add_argument("--output-json", default="")
    retrieve.set_defaults(func=command_retrieve)

    evaluate = subparsers.add_parser("eval", help="Run strict deduplicated Top-k recall")
    evaluate.add_argument("--config", default=argparse.SUPPRESS)
    evaluate.add_argument("--backends", default="", help="Comma-separated backends")
    evaluate.add_argument("--top-ks", default="", help="Comma-separated top-k values")
    evaluate.add_argument("--output-json", default="")
    evaluate.add_argument("--output-csv", default="")
    evaluate.set_defaults(func=command_eval)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
