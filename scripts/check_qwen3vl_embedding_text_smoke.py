"""Qwen3-VL-Embedding 文本编码冒烟检查。"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Qwen3-VL-Embedding 文本编码冒烟检查")
    parser.add_argument(
        "--embedder-script-path",
        default="/home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py",
    )
    parser.add_argument(
        "--embedding-model-path",
        default="/home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B",
    )
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    """将 dtype 名称解析为 torch dtype。"""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_embedder(script_path: Path, model_path: Path, torch_dtype: torch.dtype):
    """加载官方 Qwen3VLEmbedder。"""
    if not script_path.is_file():
        raise FileNotFoundError(f"Embedding 脚本不存在：{script_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Embedding 模型路径不存在：{model_path}")
    spec = importlib.util.spec_from_file_location("qwen3vl_embedding_smoke", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 embedding 脚本：{script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLEmbedder(model_name_or_path=str(model_path), torch_dtype=torch_dtype)


def main() -> None:
    """程序入口。"""
    args = parse_args()
    embedder = load_embedder(
        script_path=Path(args.embedder_script_path),
        model_path=Path(args.embedding_model_path),
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    inputs = [
        {
            "text": "low-light scene with strong backlight from the vehicle ahead",
            "instruction": "Represent the user's input.",
        },
        {
            "text": "A dashcam image caption mentioning wet road, cars, brake lights, and low visibility.",
            "instruction": "Represent the user's input.",
        },
    ]
    embeddings = embedder.process(inputs, normalize=True)
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.float().detach().cpu().numpy()
    arr = np.asarray(embeddings, dtype=np.float32)
    print({"shape": list(arr.shape), "first_norm": float(np.linalg.norm(arr[0]))})


if __name__ == "__main__":
    main()
