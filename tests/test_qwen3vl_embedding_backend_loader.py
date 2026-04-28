from __future__ import annotations

from pathlib import Path

from pipelines.retrieval_pipeline import (
    Qwen3VLEmbeddingFaissBackend,
    _resolve_migrated_image_path,
)


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


def test_resolve_migrated_defaultshare_image_path(tmp_path: Path) -> None:
    image_path = tmp_path / "input" / "raw" / "issue_1003000" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")

    resolved = _resolve_migrated_image_path(
        "/defaultShare/qwen-vl/Retriv/input/raw/issue_1003000/sample.jpg",
        project_root=tmp_path,
    )

    assert resolved == str(image_path)
