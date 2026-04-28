# Scene Retrieval Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a self-contained demo folder for image caption generation, BM25 retrieval, Qwen3-VL-Embedding retrieval, and strict deduplicated recall evaluation.

**Architecture:** The demo will live under `demo/scene_retrieval_demo/` and contain one runnable Python script plus JSON configuration, prompt configuration, query configuration, dependency notes, and a README. The script will duplicate the minimal stable logic needed for demo use, while keeping heavyweight model imports lazy so lightweight commands and tests do not require loading Qwen models.

**Tech Stack:** Python, Transformers/Qwen3.5-VL for captioning, rank_bm25 for BM25, Qwen3-VL-Embedding + FAISS for semantic retrieval, JSON/CSV outputs.

---

### Task 1: Demo Skeleton And Config

**Files:**
- Create: `demo/scene_retrieval_demo/config/demo_config.json`
- Create: `demo/scene_retrieval_demo/config/caption_prompt.json`
- Create: `demo/scene_retrieval_demo/config/scene_queries_specific_en.json`
- Create: `demo/scene_retrieval_demo/requirements-demo.txt`
- Create: `demo/scene_retrieval_demo/CODE_INVENTORY.md`

- [ ] **Step 1: Create demo config**

Create a JSON config that points at the current experiment paths, including `RetrInput_hh`, Qwen3.5 caption model path, caption output files, query file, strict eval output files, and Qwen3-VL-Embedding settings.

- [ ] **Step 2: Create prompt config**

Copy the current `SYSTEM_INSTRUCTION`, `DEFAULT_USER_QUERY`, and `FEW_SHOTS` from `src/utils/prompt_config.py` into `caption_prompt.json`.

- [ ] **Step 3: Create query config**

Copy `input/processed/scene_queries_specific_en.json` into the demo config folder.

- [ ] **Step 4: Create inventory**

Document source files used to assemble the demo: caption pipeline, batch caption script, retrieval backend, strict dedup experiment, prompt config, image loader, and query config.

### Task 2: Demo Script

**Files:**
- Create: `demo/scene_retrieval_demo/scene_retrieval_demo.py`

- [ ] **Step 1: Implement CLI**

Create subcommands:

```text
caption
retrieve
eval
```

- [ ] **Step 2: Implement caption generation**

Load prompt JSON and model path from config. Recursively collect images under `image_root`, generate captions, and write JSON/JSONL outputs.

- [ ] **Step 3: Implement retrieval**

Load caption JSON, build either BM25 or Embedding backend, run every configured query over the full caption corpus, and write ranked JSON output.

- [ ] **Step 4: Implement strict deduplicated evaluation**

Load caption JSON, deduplicate by SHA256 image content, rebuild BM25 and/or Embedding indexes over the deduplicated corpus, search all 79 unique images per query, and compute Top10/Top20/Top30 hash-level recall.

### Task 3: Tests And Docs

**Files:**
- Create: `tests/test_scene_retrieval_demo.py`
- Create: `demo/scene_retrieval_demo/README.md`

- [ ] **Step 1: Add lightweight tests**

Test config loading, top-k parsing, BM25 tokenization, and strict hash recall with fake image files and fake retrieval items.

- [ ] **Step 2: Write README**

Document what the demo does, folder contents, dependencies, model path assumptions, and exact commands for caption, BM25 retrieval, Embedding retrieval, and strict eval.

- [ ] **Step 3: Verify**

Run:

```bash
PYTHONPATH=src /home/leo494/miniforge3/envs/dc25/bin/python -m pytest -q tests/test_scene_retrieval_demo.py
PYTHONPATH=src /home/leo494/miniforge3/envs/dc25/bin/python demo/scene_retrieval_demo/scene_retrieval_demo.py --help
PYTHONPATH=src /home/leo494/miniforge3/envs/dc25/bin/python demo/scene_retrieval_demo/scene_retrieval_demo.py eval --config demo/scene_retrieval_demo/config/demo_config.json --backends bm25
```

Expected: tests pass, help renders, BM25 eval writes JSON/CSV without loading embedding model.
