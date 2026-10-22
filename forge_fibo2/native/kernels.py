# FIBO 2 support for Forge Neo
# The few Forge Neo runtime pieces the DiT calls, with plain torch fallbacks outside Forge (CPU tests).
# Same module as the FLUX 3 and MOSS-Image extensions' kernels.py.

import torch

try:
    from backend import attention as _forge_attention
    from backend.memory_management import cast_to as _forge_cast_to
except ImportError:  # outside Forge Neo: eager backend on CPU
    _forge_attention = None
    _forge_cast_to = None

__all__ = ["attention", "cast_to"]


def cast_to(weight, dtype=None, device=None):
    if _forge_cast_to is not None:
        return _forge_cast_to(weight, dtype=dtype, device=device)
    return weight.to(dtype=dtype, device=device)


def attention(q, k, v, transformer_options=None):
    """q [B, H, Sq, D], k and v [B, H, Sk, D] -> [B, Sq, H * D], no mask.

    transformer_options reach Forge's attention wrapper, which applies attention overrides such as the Sparse
    Attention Integrated extension. The torch fallback (CPU tests) pins the MATH backend, the plain reference
    implementation.
    """
    if _forge_attention is not None:
        # looked up per call: Forge picks its attention function at import time and other extensions may wrap it
        return _forge_attention.attention_function(q, k, v, q.shape[1], skip_reshape=True,
                                                   transformer_options=transformer_options or {})
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
