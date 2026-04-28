"""Issue 语料流水线：先批量图像描述，再基于 caption 语料做文本检索图像。

典型用法（单目录下全部图片，无子目录 glob）::

    cd /defaultShare/qwen-vl/Retriv && PYTHONPATH=src python \\
        src/experiments/issues_caption_then_retrieve_experiment.py \\
        --model-path /path/to/Qwen3.5-4B \\
        --image-root /defaultShare/qwen-vl/Retriv/input/raw/issue_1001000 \\
        --caption-output-json output/reports/caption_issue_1001000.json \\
        -- --top-k 10 --output-json output/reports/retrieval_issue_1001000.json

多个 ``issue_*`` 子目录（不扫到 ``example_data``）::

    cd /defaultShare/qwen-vl/Retriv && PYTHONPATH=src python \\
        src/experiments/issues_caption_then_retrieve_experiment.py \\
        --model-path /path/to/Qwen3.5-4B \\
        --image-root input/raw \\
        --subdir-glob 'issue_*' \\
        --caption-output-json output/reports/caption_issues_all.json \\
        -- --top-k 20 --output-json output/reports/retrieval_issues_all.json

``--`` 之后的参数原样传给 ``text_to_image_retrieval_experiment.py``；
本脚本会在末尾追加 ``--caption-result-json`` 指向上一步描述结果（覆盖同名字段）。
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def _retriv_src_root() -> Path:
    """定位 Retriv 仓库根目录（含 ``src/``）。"""
    return Path(__file__).resolve().parents[2]


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """按 ``--`` 拆分为描述阶段参数与检索阶段参数。"""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _strip_caption_result_json_flag(retrieval_argv: list[str]) -> list[str]:
    """移除检索参数中的 ``--caption-result-json``，由编排脚本统一注入。"""
    out: list[str] = []
    skip_next = False
    for token in retrieval_argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--caption-result-json":
            skip_next = True
            continue
        if token.startswith("--caption-result-json="):
            continue
        out.append(token)
    return out


def parse_caption_args(argv: list[str]) -> argparse.Namespace:
    """解析描述阶段命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Issue 语料：先批量描述，再检索（检索参数放在 -- 之后）",
    )
    parser.add_argument("--model-path", type=str, required=True, help="图像描述模型路径")
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="图片根目录；与 batch_image_to_text_experiment 一致",
    )
    parser.add_argument(
        "--subdir-glob",
        type=str,
        default="",
        help='若指定，仅在 image-root 下匹配的一级子目录内收集（如 "issue_*"）',
    )
    parser.add_argument("--max-images", type=int, default=0, help="最多描述张数，0 表示全部")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="描述 max_new_tokens")
    parser.add_argument(
        "--caption-output-json",
        type=str,
        required=True,
        help="描述结果 JSON 路径（相对 Retriv 根或绝对路径）",
    )
    parser.add_argument(
        "--caption-output-jsonl",
        type=str,
        default="",
        help="可选：逐行 JSONL 输出路径",
    )
    return parser.parse_args(argv)


def _default_retrieval_argv() -> list[str]:
    """未在 ``--`` 后提供检索参数时的默认最小集合。"""
    return [
        "--output-json",
        "output/reports/retrieval_after_caption_pipeline.json",
    ]


def _validate_query_file_strict(root: Path, retrieval_argv: list[str]) -> None:
    """严格校验检索参数中的 query 文件路径，不做任何兜底。"""
    i = 0
    while i < len(retrieval_argv):
        token = retrieval_argv[i]
        if token == "--query-file":
            if i + 1 >= len(retrieval_argv):
                raise ValueError("--query-file 缺少路径参数")
            path_text = retrieval_argv[i + 1]
            _assert_query_file_exists(root=root, raw_path=path_text)
            i += 2
            continue
        if token.startswith("--query-file="):
            path_text = token.split("=", 1)[1]
            _assert_query_file_exists(root=root, raw_path=path_text)
            i += 1
            continue
        i += 1


def _assert_query_file_exists(root: Path, raw_path: str) -> None:
    """断言 query 文件存在，不存在直接抛错。"""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"query 文件不存在：{candidate}")


def main() -> None:
    """顺序执行描述子进程与检索子进程。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    caption_argv, retrieval_argv = _split_argv(sys.argv[1:])
    args = parse_caption_args(caption_argv)

    root = _retriv_src_root()
    batch_py = root / "src" / "experiments" / "batch_image_to_text_experiment.py"
    retrieval_py = root / "src" / "experiments" / "text_to_image_retrieval_experiment.py"
    if not batch_py.is_file() or not retrieval_py.is_file():
        raise FileNotFoundError(f"未找到实验脚本：{batch_py} 或 {retrieval_py}")

    caption_json = Path(args.caption_output_json)
    if not caption_json.is_absolute():
        caption_json = (root / caption_json).resolve()

    env = os.environ.copy()
    src = str(root / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    batch_cmd: list[str] = [
        sys.executable,
        str(batch_py),
        "--model-path",
        args.model_path,
        "--image-root",
        args.image_root,
        "--output-json",
        str(caption_json),
        "--max-images",
        str(args.max_images),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.subdir_glob.strip():
        batch_cmd += ["--subdir-glob", args.subdir_glob.strip()]
    if args.caption_output_jsonl.strip():
        batch_cmd += ["--output-jsonl", args.caption_output_jsonl.strip()]

    LOGGER.info("Step 1: caption batch, command=%s", batch_cmd)
    subprocess.run(batch_cmd, cwd=str(root), env=env, check=True)

    ret_args = _strip_caption_result_json_flag(retrieval_argv)
    if not ret_args:
        ret_args = _default_retrieval_argv()
        LOGGER.info("No args after --; using default retrieval argv: %s", ret_args)
    _validate_query_file_strict(root=root, retrieval_argv=ret_args)
    ret_args = ret_args + ["--caption-result-json", str(caption_json)]

    retrieval_cmd: list[str] = [sys.executable, str(retrieval_py), *ret_args]
    LOGGER.info("Step 2: retrieval, command=%s", retrieval_cmd)
    subprocess.run(retrieval_cmd, cwd=str(root), env=env, check=True)
    LOGGER.info("Pipeline finished. Caption JSON: %s", caption_json)


if __name__ == "__main__":
    main()
