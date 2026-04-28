"""将用户 query 扩展为多条英文检索短语（翻译 + 相近场景）。"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

LOGGER = logging.getLogger(__name__)


def _build_expansion_prompt(user_query: str, total_phrases: int) -> str:
    """构造让模型输出 JSON 的提示词。

    Args:
        user_query: 用户原始 query（可为中文）。
        total_phrases: 需要生成的英文短语总数。

    Returns:
        str: 用户侧提示文本。
    """
    return (
        "You help build English search queries for dashcam image caption retrieval.\n\n"
        f"Input query (any language):\n{user_query}\n\n"
        "Respond with ONLY one JSON object, no markdown fences, format:\n"
        '{"english_queries": ["...", "...", ...]}\n\n'
        "Rules:\n"
        f"- The array english_queries must contain exactly {total_phrases} strings.\n"
        "- english_queries[0]: faithful English translation of the input.\n"
        f"- english_queries[1:{total_phrases}]: different short English phrases describing "
        "similar driving scenes (weather, lighting, glare, opposing traffic, brake lights, "
        "wet road, night, etc.).\n"
        "- Prefer compact noun phrases under 32 words each.\n"
        "- No duplicate strings.\n"
    )


def _parse_json_queries(raw_text: str, expected: int) -> list[str]:
    """从模型输出中解析 english_queries 列表。

    Args:
        raw_text: 模型原始输出。
        expected: 期望条数。

    Returns:
        list[str]: 英文 query 列表。

    Raises:
        ValueError: 解析失败或条数不对。
    """
    text = raw_text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and "english_queries" in part:
                text = part
                break
    data = json.loads(text)
    arr = data.get("english_queries", [])
    if not isinstance(arr, list):
        raise ValueError("JSON 中缺少 english_queries 数组")
    cleaned: list[str] = []
    seen_lower: set[str] = set()
    for x in arr:
        s = str(x).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        cleaned.append(s)
    if len(cleaned) < expected:
        raise ValueError(f"english_queries 数量不足：需要 {expected} 条，实际 {len(cleaned)} 条")
    return cleaned[:expected]


class EnglishQueryExpander:
    """使用小型多模态文本能力模型生成英文扩展 query。"""

    model_path: str
    processor: Any
    model: Any
    max_new_tokens: int

    def __init__(self, model_path: str, max_new_tokens: int = 512) -> None:
        """加载用于扩展的生成模型。

        Args:
            model_path: 模型目录。
            max_new_tokens: 生成上限。

        Raises:
            RuntimeError: 加载失败。
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            LOGGER.info("English query expander initialized: %s", model_path)
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"扩展用模型加载失败：{exc}") from exc

    def expand(self, user_query: str, total_phrases: int = 4) -> list[str]:
        """生成英文主 query 及若干相近场景短语。

        Args:
            user_query: 原始 query。
            total_phrases: 生成总数，默认 4（1 翻译 + 3 相近）。

        Returns:
            list[str]: 英文 query 列表。

        Raises:
            ValueError: 输入非法。
            RuntimeError: 生成或解析失败。
        """
        if not user_query.strip():
            raise ValueError("用户 query 不能为空")
        if total_phrases < 2:
            raise ValueError("total_phrases 至少为 2")

        prompt = _build_expansion_prompt(user_query=user_query, total_phrases=total_phrases)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.35,
                    top_p=0.9,
                )
            trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
            raw = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"扩展 query 生成失败：{exc}") from exc

        return _parse_json_queries(raw_text=raw, expected=total_phrases)
