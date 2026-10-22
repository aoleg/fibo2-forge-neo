"""Minimal stand-ins for the Forge Neo pieces the extension uses, for CPU tests without Forge.

Only the names the extension touches exist; anything else raises AttributeError, so a test notices when the
extension starts depending on more of Forge. Same shape as the FLUX 3 and MOSS-Image extensions' forge_stubs.
"""

import re
import sys
import types
from contextlib import contextmanager

import torch

# Qwen's special tokens the prompts use; every other character is one token above the special range
SPECIAL = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|vision_start|>": 151652, "<|vision_end|>": 151653,
           "<|image_pad|>": 151655}
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL) + ")")


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class FakeHFTokenizer:
    def encode(self, text, add_special_tokens=True):
        ids = []
        for part in _SPLIT.split(text):
            ids += [SPECIAL[part]] if part in SPECIAL else [1000 + ord(c) for c in part]
        return ids


class FakeSDTokenizer:
    def __init__(self, tokenizer_object=None, **kwargs):
        self.tokenizer = tokenizer_object


class FakeSDClipModel:
    """encode_token_weights as Forge's does with a layer list: states (1, L, T, D), seeded by the token count, an
    image entry standing for 4 tokens (Forge splices the vision tower's output there)."""

    layers, dim, image_tokens = 15, 8, 4

    def __init__(self, transformer=None, **kwargs):
        self.transformer = transformer
        self.kwargs = kwargs
        self.calls = []

    @classmethod
    def states(cls, length):
        generator = torch.Generator().manual_seed(length)
        return torch.randn(1, cls.layers, length, cls.dim, generator=generator) * 30

    def encode_token_weights(self, token_weight_pairs):
        pairs = token_weight_pairs[0]
        self.calls.append([pair[0] for pair in pairs])
        length = sum(self.image_tokens if isinstance(pair[0], dict) else 1 for pair in pairs)
        return self.states(length), None, {"attention_mask": torch.ones(1, length)}


def text_processing():
    """backend.args and backend.text_processing, enough to import forge_fibo2.native.text_engine."""
    comfy = module("backend.text_processing._comfy", EMBEDDINGS=list, INF=float("inf"),
                   SDTokenizer=FakeSDTokenizer, SDClipModel=FakeSDClipModel)
    emphasis = module("backend.text_processing.emphasis", uses_emphasis=lambda text: False,
                      EmphasisNone=type("EmphasisNone", (), {}))
    package = module("backend.text_processing", emphasis=emphasis, _comfy=comfy)
    args = module("backend.args", dynamic_args=types.SimpleNamespace(last_extra_generation_params={},
                                                                     ref_latents=[]))
    return {"backend": module("backend", args=args, text_processing=package), "backend.args": args,
            "backend.text_processing": package, "backend.text_processing.emphasis": emphasis,
            "backend.text_processing._comfy": comfy}


def guess(models=()):
    """huggingface_guess.model_list with the given model classes, for compat.native_support."""
    model_list = module("huggingface_guess.model_list", models=list(models))
    return {"huggingface_guess": module("huggingface_guess", model_list=model_list),
            "huggingface_guess.model_list": model_list}


@contextmanager
def installed(stubs):
    """The stubs in sys.modules for the duration, and the extension's modules re-imported against them."""
    saved = {name: sys.modules.get(name) for name in stubs}
    ours = {name: mod for name, mod in sys.modules.items() if name.startswith("forge_fibo2")}
    sys.modules.update(stubs)
    for name in ours:
        del sys.modules[name]
    try:
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        for name in [n for n in sys.modules if n.startswith("forge_fibo2")]:
            del sys.modules[name]
        sys.modules.update(ours)
