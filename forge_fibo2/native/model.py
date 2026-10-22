# FIBO 2 support for Forge Neo
# huggingface_guess registration, following the FLUX 3 extension's model.py.

import os

import torch
from huggingface_guess import latent
from huggingface_guess.model_list import BASE, ModelType

from . import detect as detection

# absolute path: the loader joins it with its own huggingface folder, and os.path.join keeps an absolute second part.
# The engine clears the loader's model flags, which it sets from substrings of this path ("klein", "Wan", "Anima"...)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "huggingface", "fibo2")
TEXT_ENCODER = "qwen3vl_4b"
# rectified flow under FLUX's exponential time shift, mu = 1.15 (ComfyUI: ModelType.FLUX, shift 1.15): Forge's
# PredictionFlux, the same 10000-step grid, so Euler + Simple gives ComfyUI's sigmas
SHIFT = 1.15


class Fibo2(BASE):
    huggingface_repo = CONFIG_DIR

    unet_config = {"image_model": "fibo2"}

    sampling_settings = {"shift": SHIFT}

    # ComfyUI's
    memory_usage_factor = 17.0
    # fp16 is allowed for GPUs without bfloat16 (RTX 20 and older): the DiT keeps its residual streams in float32
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    unet_extra_config = {}
    latent_format = latent.Flux2

    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text_encoders."]

    unet_target = "transformer"

    def clip_target(self, state_dict={}):
        # the same Qwen3-VL 4B Forge Neo loads natively for Krea 2
        return {f"{TEXT_ENCODER}.transformer": "text_encoder"}

    def model_type(self, state_dict):
        return ModelType.FLUX


def detect(state_dict: dict, key_prefix: str) -> dict | None:
    return detection.detect(state_dict, key_prefix)
