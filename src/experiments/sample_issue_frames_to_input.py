"""从 issue 的 frames 目录随机抽样图片到 Retriv/input/raw。

规则：
- 每个 issue 最多抽样 N 张（默认 20）。
- 若可用图片不足 N，则复制全部可用图片，不补齐。
- 采用固定随机种子，保证可复现。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_SUFFIXES: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从 issue_xxx/frames 随机抽样复制到 Retriv/input/raw/issue_xxx",
    )
    parser.add_argument(
        "--source-root",
        type=str,
        default="/defaultShare/data_closing/scenes/busaeb/data_mining/20260316-20260324/issues",
        help="源目录，包含 issue_* 子目录",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default="/defaultShare/qwen-vl/Retriv/input/raw",
        help="目标目录，输出 issue_* 子目录",
    )
    parser.add_argument(
        "--issue-glob",
        type=str,
        default="issue_*",
        help="issue 目录匹配模式",
    )
    parser.add_argument(
        "--sample-per-issue",
        type=int,
        default=20,
        help="每个 issue 目标抽样数；不足时复制全部，不补齐",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（固定后可复现）",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default="",
        help="采样清单输出路径；留空则写到 target-root/issues_sample_manifest.json",
    )
    return parser.parse_args()


def collect_issue_dirs(source_root: Path, issue_glob: str) -> list[Path]:
    """收集 issue 目录列表。"""
    if not source_root.exists():
        raise FileNotFoundError(f"源目录不存在：{source_root}")
    issues = [p for p in sorted(source_root.iterdir()) if p.is_dir() and p.match(issue_glob)]
    if not issues:
        raise RuntimeError(f"未找到匹配 {issue_glob!r} 的 issue 目录：{source_root}")
    return issues


def collect_images(frames_dir: Path) -> list[Path]:
    """收集 frames 目录下可用图片。"""
    return sorted(
        p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def sample_images(images: list[Path], sample_per_issue: int, rng: random.Random) -> list[Path]:
    """按规则抽样（不足则全量）。"""
    if sample_per_issue <= 0:
        return images
    if len(images) <= sample_per_issue:
        return images
    chosen = rng.sample(images, sample_per_issue)
    chosen.sort()
    return chosen


def main() -> None:
    """程序入口。"""
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path.strip()
        else target_root / "issues_sample_manifest.json"
    )

    rng = random.Random(args.seed)
    issue_dirs = collect_issue_dirs(source_root=source_root, issue_glob=args.issue_glob)

    manifest: dict[str, object] = {
        "seed": args.seed,
        "sample_per_issue": args.sample_per_issue,
        "sample_rule": "不足 sample_per_issue 时复制全部可用图片，不补齐",
        "source_root": str(source_root),
        "target_root": str(target_root),
        "issues": [],
    }

    for issue_dir in issue_dirs:
        frames_dir = issue_dir / "frames"
        if not frames_dir.is_dir():
            issue_entry = {
                "issue": issue_dir.name,
                "frames_dir": str(frames_dir),
                "candidate_count": 0,
                "sampled_count": 0,
                "status": "missing_frames_dir",
                "samples": [],
            }
            cast_issues = manifest["issues"]
            if isinstance(cast_issues, list):
                cast_issues.append(issue_entry)
            continue

        images = collect_images(frames_dir=frames_dir)
        selected = sample_images(
            images=images,
            sample_per_issue=args.sample_per_issue,
            rng=rng,
        )

        target_issue_dir = target_root / issue_dir.name
        if target_issue_dir.exists():
            shutil.rmtree(target_issue_dir)
        target_issue_dir.mkdir(parents=True, exist_ok=True)

        samples: list[dict[str, object]] = []
        for index, src in enumerate(selected, start=1):
            dst_name = f"{index:02d}_{src.name}"
            dst = target_issue_dir / dst_name
            shutil.copy2(src, dst)
            samples.append(
                {
                    "index": index,
                    "source_path": str(src),
                    "target_path": str(dst),
                    "target_name": dst_name,
                }
            )

        issue_entry = {
            "issue": issue_dir.name,
            "frames_dir": str(frames_dir),
            "candidate_count": len(images),
            "sampled_count": len(samples),
            "status": "ok",
            "samples": samples,
        }
        cast_issues = manifest["issues"]
        if isinstance(cast_issues, list):
            cast_issues.append(issue_entry)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"采样完成：共处理 {len(issue_dirs)} 个 issue，seed={args.seed}")
    print(f"清单文件：{manifest_path}")


if __name__ == "__main__":
    main()
