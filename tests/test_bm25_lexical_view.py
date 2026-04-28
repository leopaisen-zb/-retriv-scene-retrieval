from __future__ import annotations

from pipelines.retrieval_pipeline import CaptionRecord
from utils.bm25_lexical_view import build_bm25_text, build_bm25_view_rows


def test_build_bm25_text_removes_timestamp_channel_and_section_labels() -> None:
    caption = """
    Timestamp 2023-04-25 13:35:34 CH4
    **Foreground**
    Wet road with the brake lights of the lead vehicle illuminated.
    Overall Composition: city traffic scene.
    """

    text = build_bm25_text(caption)

    assert "2023" not in text
    assert "ch4" not in text
    assert "foreground" not in text
    assert "overall" not in text
    assert "wet" in text
    assert "brake" in text
    assert "city traffic scene" in text


def test_build_bm25_text_removes_caption_heading_variants_without_dropping_sentences() -> None:
    caption = """
    Lighting and Atmosphere:
    Background and Environment:
    Main Subjects and Spatial Positioning:
    Foreground & Immediate Surroundings:
    Foreground and Immediate Surroundings:
    Midground:
    Overlays:
    **Foreground**
    Overall Composition:
    The lighting is poor, but the scene is still readable.
    """

    text = build_bm25_text(caption)
    tokens = text.split()

    assert "foreground" not in tokens
    assert "background" not in tokens
    assert "immediate" not in tokens
    assert "surroundings" not in tokens
    assert "midground" not in tokens
    assert "overlays" not in tokens
    assert "atmosphere" not in tokens
    assert "environment" not in tokens
    assert "main" not in tokens
    assert "subjects" not in tokens
    assert "spatial" not in tokens
    assert "positioning" not in tokens
    assert "overall" not in tokens
    assert "lighting" in tokens
    assert "poor" in tokens
    assert "scene" in tokens


def test_build_bm25_text_removes_period_terminated_heading_only_lines() -> None:
    caption = """
    Lighting and Atmosphere.
    Overall Composition.
    The scene is still readable.
    """

    text = build_bm25_text(caption)
    tokens = text.split()

    assert "atmosphere" not in tokens
    assert "overall" not in tokens
    assert "composition" not in tokens
    assert "scene" in tokens


def test_build_bm25_text_keeps_scene_terms_and_deduplicates_tokens() -> None:
    caption = """
    Rainy road scene with wet reflective pavement.
    A lead vehicle ahead shows bright brake lights.
    The wet road reflects the brake lights.
    """

    text = build_bm25_text(caption)
    tokens = text.split()

    assert "rainy" in tokens
    assert "wet" in tokens
    assert "brake" in tokens
    assert tokens.count("wet") == 1


def test_build_bm25_text_preserves_section_words_in_normal_sentences() -> None:
    caption = "The lighting is poor, but the scene is still readable."

    text = build_bm25_text(caption)
    tokens = text.split()

    assert "lighting" in tokens
    assert "poor" in tokens
    assert "scene" in tokens


def test_build_bm25_text_removes_plate_like_tokens_without_dropping_scene_terms() -> None:
    caption = """
    Vehicle ABC1234 ahead on a wet road with bright brake lights.
    """

    text = build_bm25_text(caption)
    tokens = text.split()

    assert "abc1234" not in tokens
    assert "wet" in tokens
    assert "brake" in tokens
    assert "ahead" in tokens


def test_build_bm25_view_rows_preserves_record_identity() -> None:
    records = [
        CaptionRecord(
            image_id="img-1",
            image_path="/tmp/img-1.jpg",
            caption="Night scene with strong glare from the vehicle ahead.",
            source_result_file="captions.json",
        )
    ]

    rows = build_bm25_view_rows(records)

    assert len(rows) == 1
    assert rows[0].image_id == "img-1"
    assert rows[0].image_path == "/tmp/img-1.jpg"
    assert rows[0].source_caption == records[0].caption
    assert "glare" in rows[0].bm25_text.split()
