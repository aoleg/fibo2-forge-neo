# FIBO 2 support for Forge Neo
# The time shift of Bria's pipeline (diffusers' BriaFibo2Pipeline, Apache-2.0): FLUX's exponential shift, its mu set
# by the picture's size, linear in the token count through (256 tokens, 0.5) and (4096 tokens, 1.15), not clamped.
# A token is a 16x16 pixel patch, so 1024x1024 gives 4096 tokens and mu 1.15, the fixed shift ComfyUI uses; other
# sizes get their own (832x1216: ~1.13, 1424x800: ~1.21, 2048x2048: ~3.23).
#
# No Forge imports: the CPU tests run it as is.

BASE_SEQ_LEN, MAX_SEQ_LEN = 256, 4096
BASE_SHIFT, MAX_SHIFT = 0.5, 1.15
PATCH = 16


def calculate_shift(image_seq_len: int) -> float:
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    return image_seq_len * m + BASE_SHIFT - m * BASE_SEQ_LEN


def dynamic_shift(width: int, height: int) -> float:
    """mu for a width x height picture, as the reference pipeline sets it."""
    return calculate_shift((int(height) // PATCH) * (int(width) // PATCH))
