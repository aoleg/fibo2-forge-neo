"""Detection, file checks and LoRA keys on the real files' tensor shapes (tests/fixtures, read from the headers of
Comfy-Org/Fibo-2.0's safetensors)."""

import json
from pathlib import Path

import pytest
import torch

from forge_fibo2.native import checkpoints, detect, lora, transformer

FIXTURES = Path(__file__).parent / "fixtures"


def header(name):
    return json.loads((FIXTURES / f"{name}.shapes.json").read_text())


EXPECTED = {"image_model": "fibo2", "in_channels": 128, "dim": 3584, "n_heads": 28, "n_layers": 44,
            "n_refiner_layers": 2, "injection_layers": [6, 14, 22, 30, 38], "cap_feat_dim": 7680,
            "perceiver_layers": 2, "max_gist_tokens": 512, "ffn_dim": 9728, "fused_ffn": True}


@pytest.mark.parametrize("name", ["fibo2_dev_bf16", "fibo2_dev_int8_convrot"])
def test_detects_the_comfy_org_files(name):
    assert detect.detect(header(name)) == EXPECTED


def test_detects_under_a_prefix_and_ignores_other_models():
    prefixed = {f"model.diffusion_model.{k}": v for k, v in header("fibo2_dev_bf16").items()}
    assert detect.detect(prefixed, "model.diffusion_model.") == EXPECTED
    assert detect.detect(header("qwen3vl_4b_int8_convrot")) is None
    assert detect.detect({"perceiver.latent_queries": {"shape": [512, 3584]}}) is None


def test_the_detected_config_builds_every_weight_of_the_file():
    # on the meta device: names and shapes only
    with torch.device("meta"):
        model = transformer.Fibo2DiT(**{k: v for k, v in EXPECTED.items() if k != "image_model"})
    ours = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    theirs = {k: tuple(v["shape"]) for k, v in header("fibo2_dev_bf16").items()}
    assert ours == theirs


def test_int8_file_differs_only_by_quantization_entries():
    bf16, int8 = header("fibo2_dev_bf16"), header("fibo2_dev_int8_convrot")
    extra = set(int8) - set(bf16)
    assert set(bf16) <= set(int8)
    assert all(k.endswith((".comfy_quant", ".weight_scale", ".comfy_attention.config")) for k in extra)

    configs = [k for k in int8 if k.endswith(".comfy_attention.config")]
    cleaned = dict(int8)
    checkpoints.clean_transformer_state_dict(cleaned)
    assert configs and len(int8) - len(cleaned) == len(configs)
    assert not any(k.endswith(".comfy_attention.config") for k in cleaned)


def test_text_encoder_checks():
    te = header("qwen3vl_4b_int8_convrot")
    checkpoints.check_text_encoder(te)
    with pytest.raises(ValueError, match="vision tower"):
        checkpoints.check_text_encoder({k: v for k, v in te.items() if "visual" not in k})
    with pytest.raises(ValueError, match="needs its Qwen3-VL 4B"):
        checkpoints.check_text_encoder(header("fibo2_dev_bf16"))
    wrong = dict(te)
    wrong["model.language_model.layers.0.input_layernorm.weight"] = {"shape": [4096]}
    with pytest.raises(ValueError, match="another size"):
        checkpoints.check_text_encoder(wrong)
    with pytest.raises(ValueError, match="FLUX.2 autoencoder"):
        checkpoints.check_vae({})


class Meta:
    def __init__(self, shape):
        self.shape = shape


def model_state(name="fibo2_dev_bf16"):
    return {f"diffusion_model.{k}": Meta(v["shape"]) for k, v in header(name).items()}


def test_turbo_lora_keys_all_reach_the_model():
    # Forge's generic map names "diffusion_model.<module>"; every module of Comfy-Org's turbo LoRA must exist
    state = model_state()
    lora_keys = header("fibo_2.0_turbo_lora_avg_rank_270_bf16")
    suffixes = (".lora_up.weight", ".lora_down.weight", ".alpha", ".diff_b", ".diff")
    modules = {k[: -len(s)] for k in lora_keys for s in suffixes if k.endswith(s)}
    missing = [m for m in modules if f"{m}.weight" not in state and m not in state]
    assert not missing


def test_lora_aliases_and_w13_halves():
    keys = lora.fibo2_lora_keys(model_state())
    qkv = "diffusion_model.layers.3.attention.qkv.weight"
    assert keys["transformer.layers.3.attention.qkv"] == qkv
    assert keys["layers.3.attention.qkv"] == qkv
    assert keys["lora_unet_layers_3_attention_qkv"] == qkv
    w13 = "diffusion_model.layers.3.feed_forward.w13.weight"
    assert keys["transformer.layers.3.feed_forward.w1"] == (w13, (0, 0, 9728))
    assert keys["transformer.layers.3.feed_forward.w3"] == (w13, (0, 9728, 9728))
    assert keys["perceiver.layers.0.ffn.w3"] == ("diffusion_model.perceiver.layers.0.ffn.w13.weight", (0, 9728, 9728))
    assert keys["transformer.perceiver.latent_queries"] == "diffusion_model.perceiver.latent_queries"
