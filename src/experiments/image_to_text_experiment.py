"""单帧图片生成文本实验入口。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from pipelines.caption_pipeline import CaptionPipeline


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        argparse.Namespace: 命令行参数对象。
    """
    parser = argparse.ArgumentParser(description="图片生成文字实验入口")
    parser.add_argument("--model-path", type=str, required=True, help="模型目录或模型名称")
    parser.add_argument("--image-path", type=str, default="", help="输入图片路径")
    parser.add_argument(
        "--image-b64-file",
        type=str,
        default="",
        help="包含 base64 图片字符串的文本文件路径",
    )
    parser.add_argument(
        "--output-text",
        type=str,
        default="",
        help="纯文本输出路径，留空则仅打印到标准输出",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="JSON 输出路径，留空则不写入 JSON 文件",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="生成最大 token 数量",
    )
    return parser.parse_args()


def _load_image_b64(image_b64_file: str) -> str:
    """读取 base64 图片字符串。

    Args:
        image_b64_file: base64 文件路径。

    Returns:
        str: base64 图片内容。

    Raises:
        ValueError: 文件不存在或内容为空。
    """
    path = Path(image_b64_file)
    if not path.exists():
        raise ValueError(f"图片 base64 文件不存在：{image_b64_file}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"图片 base64 文件内容为空：{image_b64_file}")
    return content


def _build_payload(
    model_path: str,
    image_path: str,
    image_b64_file: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """构建推理输入参数。

    Args:
        model_path: 模型路径。
        image_path: 图片路径。
        image_b64_file: base64 文件路径。
        max_new_tokens: 生成 token 上限。

    Returns:
        dict[str, Any]: 推理参数字典。

    Raises:
        ValueError: 输入参数非法。
    """
    if bool(image_path) == bool(image_b64_file):
        raise ValueError("必须且只能提供一种图片输入：--image-path 或 --image-b64-file")

    payload: dict[str, Any] = {
        "model_path": model_path,
        "max_new_tokens": max_new_tokens,
    }
    if image_path:
        payload["image_path"] = image_path
    else:
        payload["image_b64"] = _load_image_b64(image_b64_file=image_b64_file)
    return payload


def main() -> None:
    """程序入口函数。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()

    try:
        payload = _build_payload(
            model_path=args.model_path,
            image_path=args.image_path,
            image_b64_file=args.image_b64_file,
            max_new_tokens=args.max_new_tokens,
        )
        pipeline = CaptionPipeline(model_path=payload["model_path"])
        result_text = pipeline.infer_one(
            image_path=payload.get("image_path", ""),
            image_b64=payload.get("image_b64", ""),
            max_new_tokens=payload["max_new_tokens"],
        )
        print(result_text)

        if args.output_text:
            Path(args.output_text).write_text(result_text, encoding="utf-8")
            LOGGER.info("Text output saved to %s", args.output_text)

        if args.output_json:
            output_payload = {
                "model_path": args.model_path,
                "image_path": args.image_path,
                "used_b64_file": args.image_b64_file,
                "max_new_tokens": args.max_new_tokens,
                "caption": result_text,
            }
            Path(args.output_json).write_text(
                json.dumps(output_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            LOGGER.info("JSON output saved to %s", args.output_json)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"实验执行失败：{exc}") from exc


if __name__ == "__main__":
    main()
