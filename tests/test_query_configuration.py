from __future__ import annotations

import ast
import json
from pathlib import Path

def _load_default_scene_queries() -> list[str]:
    source = Path("src/experiments/text_to_image_retrieval_experiment.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_SCENE_QUERIES":
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == "DEFAULT_SCENE_QUERIES":
                return ast.literal_eval(node.value)
    raise AssertionError("DEFAULT_SCENE_QUERIES not found")


def test_default_scene_queries_are_english() -> None:
    assert _load_default_scene_queries() == [
        "vehicle ahead under strong backlighting in low-light conditions",
        "oncoming vehicle approaching in the opposite lane",
        "rainy road scene with the brake lights of the vehicle ahead illuminated",
    ]


def test_scene_queries_file_uses_english_query_text() -> None:
    path = Path("input/processed/scene_queries.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["queries"] == [
        {
            "query": "vehicle ahead under strong backlighting in low-light conditions",
            "english_query": "vehicle ahead under strong backlighting in low-light conditions",
        },
        {
            "query": "oncoming vehicle approaching in the opposite lane",
            "english_query": "oncoming vehicle approaching in the opposite lane",
        },
        {
            "query": "rainy road scene with the brake lights of the vehicle ahead illuminated",
            "english_query": "rainy road scene with the brake lights of the vehicle ahead illuminated",
        },
    ]
