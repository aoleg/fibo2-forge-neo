# FIBO 2 support for Forge Neo
# FIBO 2 LoRAs on Forge's own LoRA path. The DiT keeps ComfyUI's module names, so Forge's generic key map already
# matches "diffusion_model.<layer>" (the Comfy-Org turbo LoRA's format, norms' diffs included) and
# "lora_unet_<layer>"; this adds the "transformer." prefix and bare layer names other trainers write, and, for the
# Comfy-Org files, which fuse w1 and w3 into one w13, sends w1 / w3 LoRAs to their half of it.
#
# The hook wraps model_lora_keys_unet where sd_forge_lora (Forge's built-in LoRA extension) and backend.patcher.lora
# hold it, and only adds keys for a FIBO 2 DiT; everything else gets Forge's map untouched.

import sys

HALVES = ("w1", "w3")
_hooked = False


def fibo2_lora_keys(state_dict) -> dict:
    """LoRA key -> model key, or (model key, (dim, offset, size)) for a half of a fused w13 weight."""
    key_map = {}
    for key in state_dict:
        if not key.startswith("diffusion_model."):
            continue
        layer = key[len("diffusion_model."):]
        if layer.endswith(".weight"):
            layer = layer[: -len(".weight")]
        targets = [(layer, key)]
        if layer.endswith(".w13") and key.endswith(".weight"):
            half = state_dict[key].shape[0] // 2
            base = layer[: -len(".w13")]
            targets += [(f"{base}.{name}", (key, (0, i * half, half))) for i, name in enumerate(HALVES)]
        for name, to in targets:
            for alias in (f"transformer.{name}", name, f"lycoris_{name.replace('.', '_')}"):
                key_map.setdefault(alias, to)
            key_map.setdefault(f"diffusion_model.{name}", to)
            key_map.setdefault(f"lora_unet_{name.replace('.', '_')}", to)
    return key_map


def _is_fibo2(model) -> bool:
    from .transformer import Fibo2DiT

    return isinstance(getattr(model, "diffusion_model", None), Fibo2DiT)


def _wrap(original):
    if getattr(original, "_fibo2", False):
        return original

    def model_lora_keys_unet(model, key_map=None):
        key_map = original(model, {} if key_map is None else key_map)
        if _is_fibo2(model):
            for alias, to in fibo2_lora_keys(model.state_dict()).items():
                key_map.setdefault(alias, to)
        return key_map

    model_lora_keys_unet._fibo2 = True
    return model_lora_keys_unet


def install() -> bool:
    """Wrap the key map in backend.patcher.lora and, once it is imported, in sd_forge_lora's networks."""
    global _hooked
    from backend.patcher import lora as forge_lora

    forge_lora.model_lora_keys_unet = _wrap(forge_lora.model_lora_keys_unet)
    networks = sys.modules.get("networks")
    if networks is None or not hasattr(networks, "model_lora_keys_unet"):
        return False
    networks.model_lora_keys_unet = _wrap(networks.model_lora_keys_unet)
    _hooked = True
    return True


def ensure_installed() -> None:
    if not _hooked:
        install()
