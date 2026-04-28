"""批量图片生成文本实验入口。"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import time
from pathlib import Path
from typing import Any

from pipelines.caption_pipeline import CaptionPipeline


LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 参数对象。
    """
    parser = argparse.ArgumentParser(description="批量图片生成文本实验入口")
    parser.add_argument("--model-path", type=str, required=True, help="模型目录或模型名称")
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="图片根目录；若指定 --subdir-glob，则在其下匹配的一级子目录内递归收集图片",
    )
    parser.add_argument(
        "--subdir-glob",
        type=str,
        default="",
        help='仅处理 image-root 下名称匹配该 glob 的一级子目录（如 "issue_*"）；留空则在整个 image-root 下递归收集',
    )
    parser.add_argument("--output-json", type=str, required=True, help="结果 JSON 输出路径")
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="",
        help="逐行 JSON 输出路径，留空表示不写入",
    )
    parser.add_argument("--max-images", type=int, default=0, help="最多处理图片数量，0 表示全部")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="最大生成 token 数")
    return parser.parse_args()


def collect_images(image_root: Path, max_images: int, subdir_glob: str = "") -> list[Path]:
    """收集图片路径。

    Args:
        image_root: 根目录。
        max_images: 最大图片数。
        subdir_glob: 若非空，仅在 image_root 下匹配的一级子目录内收集。

    Returns:
        list[Path]: 图片路径列表。

    Raises:
        FileNotFoundError: 根目录不存在。
        ValueError: 未匹配到任何子目录。
    """
    if not image_root.exists():
        raise FileNotFoundError(f"图片目录不存在：{image_root}")
    images: list[Path] = []
    if subdir_glob.strip():
        matched = [
            p
            for p in sorted(image_root.iterdir())
            if p.is_dir() and fnmatch.fnmatch(p.name, subdir_glob.strip())
        ]
        if not matched:
            raise ValueError(f"在 {image_root} 下未找到匹配 {subdir_glob!r} 的一级子目录")
        for sub in matched:
            for path in sub.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    images.append(path)
    else:
        for path in image_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append(path)
    images.sort()
    if max_images > 0:
        images = images[:max_images]
    return images


def main() -> None:
    """程序入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    image_root = Path(args.image_root)
    output_json = Path(args.output_json)
    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else None

    images = collect_images(
        image_root=image_root,
        max_images=args.max_images,
        subdir_glob=args.subdir_glob,
    )
    if not images:
        raise RuntimeError(f"未找到可处理图片：{image_root}")

    LOGGER.info("Total images found: %d", len(images))
    load_start = time.perf_counter()
    pipeline = CaptionPipeline(model_path=args.model_path)
    load_elapsed = time.perf_counter() - load_start
    LOGGER.info("Model loaded in %.3f seconds", load_elapsed)

    results: list[dict[str, Any]] = []
    infer_times: list[float] = []

    if output_jsonl:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(images, start=1):
        LOGGER.info("[%d/%d] Processing %s", index, len(images), image_path)
        try:
            infer_start = time.perf_counter()
            caption = pipeline.infer_one(
                image_path=str(image_path),
                max_new_tokens=args.max_new_tokens,
            )
            infer_elapsed = time.perf_counter() - infer_start
            infer_times.append(infer_elapsed)
            row = {
                "image_path": str(image_path),
                "success": True,
                "inference_seconds": round(infer_elapsed, 6),
                "caption": caption,
            }
            results.append(row)
            if output_jsonl:
                with output_jsonl.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # pylint: disable=broad-except
            results.append(
                {
                    "image_path": str(image_path),
                    "success": False,
                    "error": str(exc),
                }
            )

    avg_infer = sum(infer_times) / len(infer_times) if infer_times else None
    summary = {
        "model_path": args.model_path,
        "image_root": str(image_root),
        "subdir_glob": args.subdir_glob or None,
        "total_images": len(images),
        "success_images": len(infer_times),
        "failed_images": len(images) - len(infer_times),
        "model_load_seconds": round(load_elapsed, 6),
        "average_inference_seconds": round(avg_infer, 6) if avg_infer else None,
        "max_new_tokens": args.max_new_tokens,
    }
    output_payload = {"summary": summary, "results": results}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Batch result saved to %s", output_json)


if __name__ == "__main__":
    main()
