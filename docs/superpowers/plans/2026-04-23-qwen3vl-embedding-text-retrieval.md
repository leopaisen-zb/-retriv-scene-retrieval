# Qwen3-VL-Embedding Text Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a first Qwen3-VL-Embedding-2B text-to-text retrieval feasibility experiment over Retriv's 6-issue, 120-image caption corpus.

**Architecture:** Keep Retriv's existing caption JSON -> `CaptionRecord` -> embedding backend -> FAISS -> JSON/CSV/per-query image folder pipeline. Adapt only the Qwen3-VL-Embedding backend path handling and experiment reporting so it loads the official `Qwen3VLEmbedder` implementation and records the actual model/repo paths.

**Tech Stack:** Python, PyTorch, Transformers, qwen-vl-utils, official `QwenLM/Qwen3-VL-Embedding`, FAISS, existing Retriv CLI scripts.

---

## File Structure

- Modify: `src/pipelines/retrieval_pipeline.py`
  - Keep `Qwen3VLEmbeddingFaissBackend`.
  - Make official embedder loading more robust.
  - Store `embedder_script_path` and `embedding_model_path` on the backend for downstream reporting.
- Modify: `src/experiments/text_to_image_retrieval_experiment.py`
  - Update default `--embedder-script-path` to the official repo module path.
  - Add model and embedder path into JSON summary.
- Create: `tests/test_qwen3vl_embedding_backend_loader.py`
  - Unit-test dynamic loader behavior using a fake local `Qwen3VLEmbedder` script so the test does not require real model weights.
- Create: `scripts/check_qwen3vl_embedding_text_smoke.py`
  - Minimal smoke script that loads official Qwen3-VL-Embedding and encodes two text records.
- Modify: `docs/dev/experiment_run_record.md`
  - Append the actual commands and experiment results after execution.
- Keep generated outputs under:
  - `output/reports/`
  - `output/retrieval_output/`

This workspace is not currently a Git repository, so commit steps are replaced by local file and command verification.

## Task 1: Official Artifact And Environment Check

**Files:**
- External create/download: `/home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding`
- External create/download: `/home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B`

- [ ] **Step 1: Check expected directories**

Run:

```bash
test -d /home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding && echo repo_exists || echo repo_missing
test -d /home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B && echo model_exists || echo model_missing
```

Expected before download: one or both may print `*_missing`.

- [ ] **Step 2: Clone the official repository if missing**

Run:

```bash
mkdir -p /home/leo494/projects/Retriv/.external /home/leo494/projects/Retriv/.models
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git /home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding
```

Expected: repository contains `src/models/qwen3_vl_embedding.py`.

- [ ] **Step 3: Download model weights if missing**

Preferred command if `huggingface-cli` is available after dependency setup:

```bash
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B \
  --local-dir /home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B
```

Fallback command if HuggingFace CLI is unavailable but Git LFS is available:

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B \
  /home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B
```

Expected: model directory contains config/tokenizer files and safetensors shard files.

- [ ] **Step 4: Install or verify official dependencies**

Run in the environment used for Qwen experiments:

```bash
/home/leo494/miniforge3/envs/dc25/bin/python -m pip install -U "transformers>=4.57.0" "qwen-vl-utils>=0.0.14" "accelerate" "huggingface_hub" "faiss-cpu"
/home/leo494/miniforge3/envs/dc25/bin/python -m pip install -e /home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding
```

Expected: imports for `torch`, `transformers`, `qwen_vl_utils`, and `faiss` succeed.

## Task 2: Add Loader Unit Test

**Files:**
- Create: `tests/test_qwen3vl_embedding_backend_loader.py`

- [ ] **Step 1: Write the failing test**

Create the file with:

```python
from __future__ import annotations

from pathlib import Path

from pipelines.retrieval_pipeline import Qwen3VLEmbeddingFaissBackend


def test_load_embedder_from_official_style_python_file(tmp_path: Path) -> None:
    script_path = tmp_path / "qwen3_vl_embedding.py"
    model_path = tmp_path / "Qwen3-VL-Embedding-2B"
    model_path.mkdir()
    script_path.write_text(
        "\n".join(
            [
                "class Qwen3VLEmbedder:",
                "    def __init__(self, model_name_or_path, torch_dtype):",
                "        self.model_name_or_path = model_name_or_path",
                "        self.torch_dtype = torch_dtype",
                "    def process(self, inputs, normalize=True):",
                "        return [[1.0, 0.0] for _ in inputs]",
            ]
        ),
        encoding="utf-8",
    )

    embedder = Qwen3VLEmbeddingFaissBackend._load_embedder(
        embedder_script_path=str(script_path),
        embedding_model_path=str(model_path),
        torch_dtype="float32",
    )

    assert embedder.model_name_or_path == str(model_path)
```

- [ ] **Step 2: Run test to verify current behavior**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_qwen3vl_embedding_backend_loader.py
```

Expected: FAIL initially if pytest or dependencies are missing; otherwise it may already pass if the existing static method supports this shape. If it passes, keep it as regression coverage.

## Task 3: Make Backend Reporting And Official Path Handling Explicit

**Files:**
- Modify: `src/pipelines/retrieval_pipeline.py`

- [ ] **Step 1: Store resolved paths on the backend**

In `Qwen3VLEmbeddingFaissBackend.__init__`, after `self.records = records`, add:

```python
self.embedder_script_path = str(Path(embedder_script_path))
self.embedding_model_path = str(Path(embedding_model_path))
```

Then pass those stored values into `_load_embedder`.

- [ ] **Step 2: Keep the dynamic loader strict and clear**

Ensure `_load_embedder` continues to:

```python
script_path = Path(embedder_script_path)
if not script_path.exists():
    raise RuntimeError(f"Embedding 脚本不存在：{embedder_script_path}")
if not Path(embedding_model_path).exists():
    raise RuntimeError(f"Embedding 模型路径不存在：{embedding_model_path}")
```

The expected embedder class remains `Qwen3VLEmbedder`.

- [ ] **Step 3: Run loader test**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_qwen3vl_embedding_backend_loader.py
```

Expected: PASS.

## Task 4: Update Retrieval CLI Defaults And Summary

**Files:**
- Modify: `src/experiments/text_to_image_retrieval_experiment.py`

- [ ] **Step 1: Update the default embedder path**

Change `--embedder-script-path` default to:

```python
default="/home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py"
```

- [ ] **Step 2: Add model paths to output summary**

After existing summary fields are populated, add:

```python
summary_block["embedder_script_path"] = (
    args.embedder_script_path if args.backend == "qwen3vl_embedding_faiss" else None
)
summary_block["embedding_model_path"] = (
    args.embedding_model_path if args.backend == "qwen3vl_embedding_faiss" else None
)
summary_block["embedding_instruction"] = (
    args.embedding_instruction if args.backend == "qwen3vl_embedding_faiss" else None
)
```

- [ ] **Step 3: Run parser-level smoke command**

Run:

```bash
PYTHONPATH=src /home/leo494/miniforge3/envs/dc25/bin/python src/experiments/text_to_image_retrieval_experiment.py --help >/tmp/retriv_text_retrieval_help.txt
rg "embedder-script-path|embedding-model-path|no-expand-english-queries" /tmp/retriv_text_retrieval_help.txt
```

Expected: command exits 0 and prints the three option names.

## Task 5: Add Real Model Smoke Script

**Files:**
- Create: `scripts/check_qwen3vl_embedding_text_smoke.py`

- [ ] **Step 1: Create the script**

Create:

```python
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
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
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_embedder(script_path: Path, model_path: Path, torch_dtype: torch.dtype):
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
    args = parse_args()
    embedder = load_embedder(
        script_path=Path(args.embedder_script_path),
        model_path=Path(args.embedding_model_path),
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    inputs = [
        {"text": "low-light scene with strong backlight from the vehicle ahead", "instruction": "Represent the user's input."},
        {"text": "A dashcam image caption mentioning wet road, cars, brake lights, and low visibility.", "instruction": "Represent the user's input."},
    ]
    embeddings = embedder.process(inputs, normalize=True)
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.float().detach().cpu().numpy()
    arr = np.asarray(embeddings, dtype=np.float32)
    print({"shape": list(arr.shape), "first_norm": float(np.linalg.norm(arr[0]))})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke script**

Run:

```bash
PYTHONPATH=src /home/leo494/miniforge3/envs/dc25/bin/python scripts/check_qwen3vl_embedding_text_smoke.py
```

Expected: exits 0 and prints a shape like `{"shape": [2, N], "first_norm": 1.0...}`.

## Task 6: Validate 120-Image Caption Corpus

**Files:**
- Read: `input/raw/issues_sample_manifest.json`
- Read: `output/reports/caption_issues_all_qwen35.json`

- [ ] **Step 1: Count input samples**

Run:

```bash
python3 -c 'import json; d=json.load(open("input/raw/issues_sample_manifest.json", encoding="utf-8")); print(sum(int(x.get("sampled_count", 0)) for x in d.get("issues", [])))'
```

Expected: `120`.

- [ ] **Step 2: Count successful caption records**

Run:

```bash
python3 -c 'import json; d=json.load(open("output/reports/caption_issues_all_qwen35.json", encoding="utf-8")); rows=d.get("results", []); print(len(rows), sum(1 for r in rows if r.get("success")))'
```

Expected: `120 120`.

- [ ] **Step 3: If caption records are fewer than 120, regenerate captions**

Run:

```bash
PYTHONPATH=src /home/leo494/miniforge3/envs/dc25/bin/python src/experiments/batch_image_to_text_experiment.py \
  --model-path .models/Qwen3.5-4B \
  --image-root input/raw \
  --subdir-glob 'issue_*' \
  --output-json output/reports/caption_issues_all_qwen35.json \
  --output-jsonl output/reports/caption_issues_all_qwen35.jsonl \
  --max-images 0 \
  --max-new-tokens 2048
```

Expected after rerun: caption JSON summary reports 120 total and 120 success. If the input manifest itself is not 120, fix sampling data before rerunning retrieval.

## Task 7: Run Qwen3-VL-Embedding Retrieval Experiment

**Files:**
- Read: `input/processed/scene_queries.json`
- Read: `output/reports/caption_issues_all_qwen35.json`
- Write: `output/reports/retrieval_issues_all_qwen3vl_embedding.json`
- Write: `output/reports/retrieval_issues_all_qwen3vl_embedding_recall_summary.csv`
- Write: `output/retrieval_output/<query>/`

- [ ] **Step 1: Run retrieval**

Run:

```bash
PYTHONPATH=src /home/leo494/miniforge3/envs/dc25/bin/python src/experiments/text_to_image_retrieval_experiment.py \
  --backend qwen3vl_embedding_faiss \
  --embedder-script-path /home/leo494/projects/Retriv/.external/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py \
  --embedding-model-path /home/leo494/projects/Retriv/.models/Qwen3-VL-Embedding-2B \
  --query-file input/processed/scene_queries.json \
  --caption-result-json output/reports/caption_issues_all_qwen35.json \
  --top-k 20 \
  --min-score 0.2 \
  --no-expand-english-queries \
  --output-json output/reports/retrieval_issues_all_qwen3vl_embedding.json
```

Expected: JSON, CSV, and per-query folders are written. Each query should return up to 20 rows.

- [ ] **Step 2: Summarize results by issue**

Run:

```bash
python3 -c 'import json, collections; d=json.load(open("output/reports/retrieval_issues_all_qwen3vl_embedding.json", encoding="utf-8")); print(json.dumps(d["summary"], ensure_ascii=False, indent=2)); [print(r["query"], collections.Counter(p.split("/issue_")[1].split("/")[0] for p in r["recalled_image_paths"])) for r in d["query_recall_summary"]]'
```

Expected: prints summary and one issue-distribution counter per query.

## Task 8: Review And Record Experiment Result

**Files:**
- Modify: `docs/dev/experiment_run_record.md`

- [ ] **Step 1: Append a dated result section**

Append a section titled:

```markdown
## 2026-04-23 Qwen3-VL-Embedding-2B 文搜文可行性实验
```

Include:
- official repo path
- model path
- caption corpus count
- retrieval command
- query count
- per-query returned count
- per-query score range
- per-query issue distribution
- conclusion: feasible / blocked / inconclusive

- [ ] **Step 2: Final verification**

Run:

```bash
rg -n "Qwen3-VL-Embedding-2B 文搜文可行性实验|corpus_image_count|embedder_script_path|embedding_model_path" docs/dev/experiment_run_record.md output/reports/retrieval_issues_all_qwen3vl_embedding.json
```

Expected: lines exist in both docs and output JSON.

## Self-Review

Spec coverage:
- Official repo and weight acquisition: Task 1.
- Backend adaptation: Tasks 2-4.
- Smoke validation: Task 5.
- 120-image corpus validation: Task 6.
- Retrieval experiment: Task 7.
- Review and result recording: Task 8.

Placeholder scan:
- No `TBD`, `TODO`, or unspecified implementation steps are left.

Type consistency:
- The backend name remains `Qwen3VLEmbeddingFaissBackend`.
- The expected official class remains `Qwen3VLEmbedder`.
- CLI flags match existing names: `--embedder-script-path`, `--embedding-model-path`, `--no-expand-english-queries`.
