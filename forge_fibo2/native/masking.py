# FIBO 2 support for Forge Neo
# Masked edits. Bria's pipeline (diffusers' BriaFibo2Pipeline, Apache-2.0, image_processor.grey_out) greys out the
# region a mask marks in the one picture to edit before it is encoded, for the VAE and for Qwen3-VL alike: a
# (128, 128, 128) grey through the mask made binary at 128. Here the mask is img2img's inpaint mask, as Forge has
# prepared it for the picture (inverted, cropped for "Only masked", resized), handed over by the extension's script
# (scripts/fibo2.py) and kept until the next generation starts.
#
# No Forge imports: the CPU tests run it as is.

import numpy as np
import torch

GREY = 128 / 255
THRESHOLD = 128

_mask = None


def set_mask(mask) -> None:
    """The inpaint mask of the generation starting (a PIL image, white where the picture changes), or None."""
    global _mask
    _mask = mask


def current():
    return _mask


def grey_out(image: torch.Tensor, mask) -> torch.Tensor:
    """image (1, H, W, 3) in [0, 1] with the region the mask marks, white on black, grey; the mask is brought to the
    picture's size first."""
    from PIL import Image

    height, width = image.shape[1], image.shape[2]
    mask = mask.convert("L")
    if mask.size != (width, height):
        mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
    region = torch.from_numpy(np.array(mask) >= THRESHOLD)[None, :, :, None]
    return torch.where(region, torch.full_like(image, GREY), image)
