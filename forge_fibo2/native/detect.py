# FIBO 2 support for Forge Neo
# Recognize a FIBO 2.0 DiT and read its configuration from tensor shapes.
#
# Works on a loaded state dict or on a safetensors header: shapes are read through shape_of, so no tensor has to be
# materialized. Same identity keys and readings as ComfyUI's model_detection (COMFYUI_COMMIT in native/__init__.py).

import re

IDENTITY_KEYS = ("perceiver.latent_queries", "layers.0.attention.qkv.weight", "final_layer.linear.weight")
FFN_KEYS = ("layers.0.feed_forward.w13.weight", "layers.0.feed_forward.w1.weight")
HEAD_DIM = 128
# one text bundle: 3 Qwen3-VL 4B layers of 2560 (text_engine.BUNDLE_LAYERS); read from the configuration, as
# ComfyUI does, not from weight columns that a quantized build may pack
CAP_FEAT_DIM = 7680


def shape_of(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    if isinstance(value, dict):
        return tuple(value["shape"])
    return tuple(value.shape)


def count_blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    found = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    return max(found) + 1 if found else 0


def is_fibo2(state_dict, key_prefix=""):
    return (all(f"{key_prefix}{k}" in state_dict for k in IDENTITY_KEYS)
            and any(f"{key_prefix}{k}" in state_dict for k in FFN_KEYS))


def detect(state_dict, key_prefix=""):
    """The Fibo2DiT keyword arguments for this checkpoint, or None when it is not a FIBO 2.0 DiT."""
    if not is_fibo2(state_dict, key_prefix):
        return None

    def shape(key):
        return shape_of(state_dict[f"{key_prefix}{key}"])

    keys = list(state_dict.keys())
    max_gist, dim = shape("perceiver.latent_queries")
    n_layers = count_blocks(keys, f"{key_prefix}layers.")
    fused = f"{key_prefix}{FFN_KEYS[0]}" in state_dict
    return {
        "image_model": "fibo2",
        "in_channels": shape("final_layer.linear.weight")[0],
        "dim": dim,
        "n_heads": dim // HEAD_DIM,
        "n_layers": n_layers,
        "n_refiner_layers": count_blocks(keys, f"{key_prefix}noise_refiner."),
        "injection_layers": [i for i in range(n_layers) if f"{key_prefix}layers.{i}.cross_attn_gate" in state_dict],
        "cap_feat_dim": CAP_FEAT_DIM,
        "perceiver_layers": count_blocks(keys, f"{key_prefix}perceiver.layers."),
        "max_gist_tokens": max_gist,
        # w13 carries the SwiGLU gate and up halves; the Comfy-Org files pad the width to 9728 for int8 kernels
        "ffn_dim": shape(FFN_KEYS[0])[0] // 2 if fused else shape(FFN_KEYS[1])[0],
        "fused_ffn": fused,
    }
