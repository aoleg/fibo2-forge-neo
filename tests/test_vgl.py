"""Captions whose braces sd-dynamic-prompts stripped, rebuilt (vgl.py)."""

import json

import pytest

from forge_fibo2.native.vgl import restore_caption_json

CAPTION = {
    "short_description": "A bakery sign at golden hour, with a chalkboard that reads \"FRESH BREAD\": {not a brace}",
    "objects": [
        {"description": "A green wooden sign", "location": "upper center", "relative_size": "large"},
        {"description": "A chalkboard easel, [black]", "location": "bottom right", "relative_size": "medium"},
    ],
    "background_setting": "A cobblestone street",
    "lighting": {"conditions": "golden hour", "direction": "side-lit from the left", "shadows": "long, soft"},
    "aesthetics": {"composition": "centered", "color_scheme": "warm", "mood_atmosphere": "cozy"},
    "photographic_characteristics": {"depth_of_field": "medium", "camera_angle": "eye-level"},
    "style_medium": "photograph",
    "artistic_style": "realistic",
    "text_render": [{"text": "FIBO 2 ON FORGE NEO", "location": "on the sign"}, {"text": "FRESH BREAD"}],
}


def strip(text: str) -> str:
    # what sd-dynamic-prompts does to a prompt without variants: every {x} becomes x
    return text.replace("{", "").replace("}", "")


@pytest.mark.parametrize("indent", [None, 2])
def test_rebuilds_a_caption(indent):
    text = json.dumps(CAPTION, indent=indent)
    assert json.loads(restore_caption_json(strip(text))) == {
        **CAPTION, "short_description": CAPTION["short_description"].replace("{", "").replace("}", "")}


def test_compact_caption_comes_back_identical():
    caption = {k: v for k, v in CAPTION.items() if k != "short_description"}
    text = json.dumps(caption, separators=(",", ":"))
    assert restore_caption_json(strip(text)) == text


def test_rebuilds_an_edit_with_its_structured_caption():
    edit = {"image_1": "x", "image_2": "y",
            "structured_caption": {"short_description": "A shop", "lighting": {"conditions": "dusk"},
                                   "text_render": [{"text": "SALE"}]},
            "edit_instruction": "add a sign"}
    for order in (edit, dict(reversed(list(edit.items())))):
        text = json.dumps(order)
        assert json.loads(restore_caption_json(strip(text))) == order


def test_empty_objects_come_back():
    caption = {"short_description": "A wall", "lighting": {}, "aesthetics": {}}
    assert json.loads(restore_caption_json(strip(json.dumps(caption)))) == caption


def test_other_prompts_are_left_alone():
    for text in ("a fox in the snow", json.dumps(CAPTION), '"a": "b"', '"short_description": "x" and more text',
                 '"objects": ["description": "x"', ""):
        assert restore_caption_json(text) == text
