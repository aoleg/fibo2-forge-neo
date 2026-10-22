"""Masked edits: the region the inpaint mask marks greyed out (masking.py)."""

import numpy as np
import torch
from PIL import Image

from forge_fibo2.native import masking


def test_grey_out_through_the_binary_mask():
    image = torch.rand(1, 32, 48, 3)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:16, 10:20] = 255
    mask[20, 30] = 127  # under the threshold: kept
    mask[21, 31] = 128  # at it: greyed
    out = masking.grey_out(image, Image.fromarray(mask))
    grey = torch.full((3,), 128 / 255)
    assert torch.equal(out[0, 8:16, 10:20], grey.expand(8, 10, 3))
    assert torch.equal(out[0, 21, 31], grey)
    assert torch.equal(out[0, 20, 30], image[0, 20, 30])
    assert torch.equal(out[0, :8], image[0, :8])


def test_mask_of_another_size_is_brought_to_the_picture():
    image = torch.zeros(1, 32, 32, 3)
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (0, 0, 32, 64))  # left half
    out = masking.grey_out(image, mask)
    assert (out[0, :, :16] > 0).all() and (out[0, :, 16:] == 0).all()


def test_state():
    masking.set_mask(None)
    assert masking.current() is None
    mask = Image.new("L", (8, 8))
    masking.set_mask(mask)
    assert masking.current() is mask
    masking.set_mask(None)
