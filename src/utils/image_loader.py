"""图像输入加载工具。"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def b64_to_pil(image_b64: str) -> Image.Image:
    """将 base64 字符串转换为 RGB PIL 图片。

    Args:
        image_b64: 图片 base64 字符串。

    Returns:
        Image.Image: RGB 图像对象。

    Raises:
        ValueError: 输入为空或解码失败。
    """
    if not image_b64.strip():
        raise ValueError("图片 base64 字符串为空")
    try:
        return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"图片 base64 解码失败：{exc}") from exc


def path_to_pil(image_path: str) -> Image.Image:
    """将本地图片路径加载为 RGB PIL 图片。

    Args:
        image_path: 图片路径。

    Returns:
        Image.Image: RGB 图像对象。

    Raises:
        ValueError: 路径不存在或读取失败。
    """
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"图片路径不存在：{image_path}")
    try:
        return Image.open(path).convert("RGB")
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"图片读取失败：{exc}") from exc
