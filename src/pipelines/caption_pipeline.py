"""图片到文本描述流水线。"""

from __future__ import annotations

import logging
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from utils.image_loader import b64_to_pil, path_to_pil
from utils.prompt_config import DEFAULT_USER_QUERY, FEW_SHOTS, SYSTEM_INSTRUCTION


LOGGER = logging.getLogger(__name__)


class CaptionPipeline:
    """单帧图片描述流水线。"""

    model_path: str
    processor: AutoProcessor
    model: Any

    def __init__(self, model_path: str) -> None:
        """初始化模型与处理器。

        Args:
            model_path: 模型目录或模型名称。

        Raises:
            RuntimeError: 初始化失败。
        """
        self.model_path = model_path
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            LOGGER.info("Caption pipeline initialized")
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"模型初始化失败：{exc}") from exc

    def infer_one(
        self,
        image_path: str = "",
        image_b64: str = "",
        user_query: str = DEFAULT_USER_QUERY,
        max_new_tokens: int = 2048,
    ) -> str:
        """对单张图片执行描述生成。

        Args:
            image_path: 本地图片路径。
            image_b64: 图片 base64 字符串。
            user_query: 用户提问文本。
            max_new_tokens: 最大生成 token 数。

        Returns:
            str: 生成结果文本。

        Raises:
            ValueError: 输入参数不合法。
            RuntimeError: 推理过程失败。
        """
        if bool(image_path) == bool(image_b64):
            raise ValueError("必须且只能提供一种图片输入：image_path 或 image_b64")

        pil_image = path_to_pil(image_path=image_path) if image_path else b64_to_pil(image_b64=image_b64)
        messages = self._build_messages(pil_image=pil_image, user_query=user_query)
        text = self._apply_template(messages=messages)

        try:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            # device_map="auto" 下，使用模型首参数所在设备，避免硬编码 cuda。
            model_device = next(self.model.parameters()).device
            inputs = {key: value.to(model_device) for key, value in inputs.items()}

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], output_ids, strict=True)
            ]
            decoded = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return decoded
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"图片描述生成失败：{exc}") from exc

    def _apply_template(self, messages: list[dict[str, Any]]) -> str:
        """应用 chat template，兼容不同处理器版本。

        Args:
            messages: 对话消息列表。

        Returns:
            str: 模板化文本。
        """
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @staticmethod
    def _build_messages(pil_image: Any, user_query: str) -> list[dict[str, Any]]:
        """构造模型消息格式。

        Args:
            pil_image: PIL 图像对象。
            user_query: 用户提问。

        Returns:
            list[dict[str, Any]]: 消息列表。
        """
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        messages.extend(FEW_SHOTS)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_query},
                ],
            }
        )
        return messages
