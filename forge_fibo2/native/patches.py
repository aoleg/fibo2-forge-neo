# FIBO 2 support for Forge Neo
# Runtime hooks into Forge Neo's loader and sampler. Every hook hands anything that is not FIBO 2 to the original
# function untouched; Forge Neo itself stays unchanged. Follows the FLUX 3 extension's patches.py.

import functools
import logging

import torch
from backend import loader
from backend.nn import krea
from huggingface_guess import detection, model_list

from . import model
from .checkpoints import check_text_encoder, check_vae, clean_transformer_state_dict
from .detect import CAP_FEAT_DIM
from .engine import Fibo2Engine
from .transformer import TEXT_BUNDLES, Fibo2DiT

logger = logging.getLogger("forge_fibo2")

# the conditioning's width: the text bundles and the phase marker (phases.py)
CONTEXT_DIM = TEXT_BUNDLES * CAP_FEAT_DIM + 1

_applied = False


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = model.detect(state_dict, key_prefix)
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_components() -> None:
    original = loader.load_huggingface_component

    # wraps leaves __wrapped__ behind, which inspect.getsource follows: the compat checks of the other extensions
    # that read Forge's loader source still reach it through this hook
    @functools.wraps(original)
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if not isinstance(guess, model.Fibo2):
            return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

        if cls_name == "Fibo2Transformer2DModel":
            if isinstance(state_dict, dict):
                clean_transformer_state_dict(state_dict)
            # the Krea 2 branch is Forge's generic DiT load: dtype, quantization (INT8 ConvRot, GGUF) and device
            # handling; it imports backend.nn.krea.SingleStreamDiT inside the function, so the swap routes the
            # build to us
            original_krea = krea.SingleStreamDiT
            krea.SingleStreamDiT = Fibo2DiT
            try:
                return original(guess, component_name, lib_name, "Krea2Transformer2DModel", repo_path, state_dict)
            finally:
                krea.SingleStreamDiT = original_krea
        if cls_name == "Qwen3VLModel":
            # Forge loads the 4B natively for Krea 2; the hook only explains a missing or wrong text encoder
            check_text_encoder(state_dict)
        if cls_name == "AutoencoderKLFlux2":
            check_vae(state_dict)

        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def _hook_cond_concat() -> None:
    # Forge batches the positive and negative prompts together by repeating the shorter text up to a common length;
    # the FIBO 2 gist count follows the true text length and the DiT reads every text token, so a repeated prompt is
    # a different prompt. Conditionings of different lengths run separately instead (ComfyUI does the same). Only
    # FIBO 2 conditionings are this wide.
    from backend.sampling.condition import ConditionCrossAttn

    original = ConditionCrossAttn.can_concat

    def can_concat(self, other):
        a, b = self.cond.shape, other.cond.shape
        if a[-1] == CONTEXT_DIM and a != b:
            return False
        return original(self, other)

    ConditionCrossAttn.can_concat = can_concat


def _hook_references() -> None:
    # ImageStitch (Forge's multi-reference input) only runs for the models dynamic_args.can_reference names, keyed on
    # loader flags; FIBO 2 has none
    from backend.args import dynamic_args

    original = dynamic_args.can_reference

    def can_reference(cls) -> bool:
        if original():
            return True
        from modules import shared

        return bool(getattr(shared.sd_model, "is_fibo2", False))

    dynamic_args.can_reference = classmethod(can_reference)


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if model.Fibo2 not in model_list.models:
        model_list.models.append(model.Fibo2)
    if Fibo2Engine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, Fibo2Engine)
    _hook_components()
    _hook_cond_concat()
    _hook_references()
    try:
        from . import settings

        settings.register()
    except Exception as e:
        # the img2img image is then always an image to edit
        logger.warning(f"[FIBO 2] could not add its setting: {e}")
    # Forge's built-in LoRA extension loads before this one; when it does not, the engine retries at model load
    from . import lora

    lora.install()
    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works: Euler, Simple, 30 steps at CFG 5 (Dev) or 4 at CFG 1 (Edit Turbo) by hand
        logger.warning(f"[FIBO 2] could not add the fibo2 UI presets: {e}")

    _applied = True
    print(f"[FIBO 2] native backend enabled (torch {torch.__version__})")
