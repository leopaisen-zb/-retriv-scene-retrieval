from __future__ import annotations

from dataclasses import dataclass
import re

from pipelines.retrieval_pipeline import CaptionRecord

_TIMESTAMP_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b")
_CHANNEL_RE = re.compile(r"\bch\d+\b", re.IGNORECASE)
_PLATE_RE = re.compile(r"\b[\u4e00-\u9fff]?[A-Z]{1,3}[A-Z0-9]*\d[A-Z0-9]*\b", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")
_HEADING_TERM_PATTERN = "|".join(
    (
        "atmosphere",
        "background",
        "composition",
        "details",
        "environment",
        "foreground",
        "immediate",
        "lighting",
        "main",
        "mid-?ground",
        "overlays",
        "overall",
        "positioning",
        "specific",
        "spatial",
        "subjects",
        "surroundings",
    )
)
_HEADING_PREFIX_RE = re.compile(
    rf"""
    ^\s*
    (?:\#{{1,6}}\s+|[-*+•]\s+|\d+[.)]\s+)?
    (?:\*\*|__|\*|_)?
    (?:
        (?:{_HEADING_TERM_PATTERN})
        (?:\s+(?:{_HEADING_TERM_PATTERN}|and|of|&))*
    )
    (?:\*\*|__|\*|_)?
    [^\S\r\n]*
    (?:[:：\-—–.][^\S\r\n]*|$)
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "with",
    "and",
    "from",
    "into",
    "over",
    "under",
    "image",
}


@dataclass(frozen=True)
class BM25LexicalViewRow:
    image_id: str
    image_path: str
    source_caption: str
    bm25_text: str


def _normalize_caption(source_caption: str) -> str:
    text = source_caption.lower()
    text = _TIMESTAMP_RE.sub(" ", text)
    text = _CHANNEL_RE.sub(" ", text)
    text = _PLATE_RE.sub(" ", text)
    text = _HEADING_PREFIX_RE.sub(" ", text)
    text = text.replace("*", " ")
    return re.sub(r"\s+", " ", text).strip()


def build_bm25_text(source_caption: str) -> str:
    """Build a conservative lexical-support view with deduplicated tokens."""
    cleaned = _normalize_caption(source_caption)
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(cleaned):
        if token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return " ".join(tokens)


def build_bm25_view_rows(records: list[CaptionRecord]) -> list[BM25LexicalViewRow]:
    return [
        BM25LexicalViewRow(
            image_id=record.image_id,
            image_path=record.image_path,
            source_caption=record.caption,
            bm25_text=build_bm25_text(record.caption),
        )
        for record in records
    ]
