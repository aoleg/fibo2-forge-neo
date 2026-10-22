"""The text engine on Forge stand-ins (tests/forge_stubs.py): taps, bundles, pictures and phases."""

import json

import pytest
import torch
import torch.nn.functional as F

from tests import forge_stubs


@pytest.fixture
def engine():
    with forge_stubs.installed(forge_stubs.text_processing()):
        from forge_fibo2.native import text_engine

        yield text_engine, text_engine.Fibo2TextEngine(text_encoder=None, tokenizer=forge_stubs.FakeHFTokenizer())


def expected(text_engine, length):
    states = F.rms_norm(forge_stubs.FakeSDClipModel.states(length), (forge_stubs.FakeSDClipModel.dim,), eps=1e-6)
    return states.to(torch.bfloat16)[0, text_engine.BUNDLE_INDEX].transpose(0, 1).flatten(1)


def test_taps_and_bundles(engine):
    text_engine, _ = engine
    assert text_engine.TAP_LAYERS == [5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35]
    assert [text_engine.TAP_LAYERS[i] for i in text_engine.BUNDLE_INDEX[:3]] == [9, 20, 31]
    assert len(text_engine.BUNDLE_INDEX) == 18


def test_plain_prompt(engine):
    text_engine, te = engine
    cond = te(["  a fox  "])[0]
    ids = te.token_ids("<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n")
    assert te.text_encoder.calls[-1] == ids
    assert cond.dtype == torch.bfloat16
    assert cond.shape == (len(ids), 18 * forge_stubs.FakeSDClipModel.dim + 1)
    assert torch.equal(cond[:, :-1], expected(text_engine, len(ids)))
    assert (cond[:, -1] == 0).all()


def test_text_render_packs_two_phases(engine):
    _, te = engine
    prompt = json.dumps({"short_description": "A sign", "text_render": [{"text": "OPEN"}]})
    cond = te([prompt])[0]
    late, early = te.text_encoder.calls[0], te.text_encoder.calls[1]
    # with the picture: one image entry where the marker's <|image_pad|> was
    assert sum(isinstance(t, dict) for t in late) == 1 and not any(isinstance(t, dict) for t in early)
    picture = next(t for t in late if isinstance(t, dict))
    assert picture["type"] == "image" and picture["data"].shape[1] % 32 == 0
    image_tokens = forge_stubs.FakeSDClipModel.image_tokens
    late_length = len(late) - 1 + image_tokens
    assert cond.shape[0] == len(early) + late_length
    assert cond[: len(early), -1].eq(1).all() and cond[len(early):, -1].eq(2).all()


def test_edit_images_go_first_and_negative_stays_text(engine):
    _, te = engine
    images = [torch.rand(1, 64, 96, 3), torch.rand(1, 1000, 800, 3)]
    prompt = json.dumps({"edit_instruction": "add a sign", "text_render": [{"text": "SALE"},
                                                                           {"text": "x"}],
                         "short_description": "A shop"})
    cond = te([prompt], images=images)[0]
    tokens = te.text_encoder.calls[-1]
    pictures = [t["data"] for t in tokens if isinstance(t, dict)]
    assert len(pictures) == 4
    # the images to edit (the second one thumbnailed to 768 a side), then the text pictures
    assert pictures[0].shape == (1, 64, 96, 3)
    assert pictures[1].shape == (1, 768, 608, 3)
    # with images, a single phase: the text pictures from the start
    assert len(te.text_encoder.calls) == 1 and (cond[:, -1] == 0).all()

    te([""])
    assert not any(isinstance(t, dict) for t in te.text_encoder.calls[-1])


def test_prompt_length_on_ui(engine):
    _, te = engine
    from forge_fibo2.native import caption

    assert te.tokenize("a fox") == len(te.token_ids("<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n"))
    assert te.tokenize(["", "a"])[0] == len(te.token_ids(caption.NULL_CAPTION))


def test_native_support_is_found():
    class Native:
        unet_config = {"image_model": "fibo2"}

    class Other:
        unet_config = {"image_model": "krea2"}

    with forge_stubs.installed(forge_stubs.guess([Other])):
        from forge_fibo2.native import compat

        assert compat.native_support() is None
    with forge_stubs.installed(forge_stubs.guess([Other, Native])):
        from forge_fibo2.native import compat

        assert compat.native_support() == "Native"
