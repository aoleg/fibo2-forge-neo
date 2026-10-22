"""The FIBO 2 DiT against a transcription of ComfyUI's (tests/reference/fibo2_comfy.py), on a small random model."""

import pytest
import torch

from forge_fibo2.native import phases, transformer
from tests.reference import fibo2_comfy as reference

# head width 128, as the rope axes (32 + 48 + 48) need; everything else small
CONFIG = dict(in_channels=8, dim=256, n_heads=2, n_layers=4, n_refiner_layers=2, injection_layers=(1, 3),
              cap_feat_dim=48, perceiver_layers=2, max_gist_tokens=320, ffn_dim=96, norm_eps=1e-5)
# one bundle for the Perceiver and one per injection block
TEXT_DIM = (1 + len(CONFIG["injection_layers"])) * CONFIG["cap_feat_dim"]


def models(**overrides):
    config = {**CONFIG, **overrides}
    torch.manual_seed(0)
    upstream = reference.Fibo2(**config)
    with torch.no_grad():
        for name, param in upstream.named_parameters():
            if "norm" in name:
                param.uniform_(0.5, 1.5)
            elif name.endswith("cross_attn_gate"):
                param.fill_(0.7)
            else:
                param.normal_(0.0, 0.05)
    ours = transformer.Fibo2DiT(**config)
    ours.load_state_dict(upstream.state_dict(), strict=True)
    return upstream.eval(), ours.eval()


def inputs(batch=2, h=4, w=6, text=40, seed=1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, CONFIG["in_channels"], h, w, generator=generator)
    context = torch.randn(batch, text, TEXT_DIM, generator=generator)
    return x, context


@torch.no_grad()
def test_matches_comfyui():
    upstream, ours = models()
    x, context = inputs()
    sigma = torch.tensor([0.7, 0.7])
    expected = upstream(x, sigma, context)
    torch.testing.assert_close(ours.run(x, sigma, context), expected, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_matches_comfyui_with_references():
    upstream, ours = models()
    x, context = inputs()
    generator = torch.Generator().manual_seed(2)
    refs = [torch.randn(1, CONFIG["in_channels"], 4, 6, generator=generator),
            torch.randn(1, CONFIG["in_channels"], 3, 5, generator=generator)]
    sigma = torch.tensor([0.4, 0.4])
    expected = upstream(x, sigma, context, ref_latents=refs)
    torch.testing.assert_close(ours.run(x, sigma, context, refs), expected, rtol=1e-4, atol=1e-5)


@torch.no_grad()
def test_matches_comfyui_unfused_and_long_text():
    # w1 / w3 apart, and a text long enough for more gist tokens (300 tokens -> 320)
    upstream, ours = models(fused_ffn=False)
    x, context = inputs(batch=1, text=300)
    sigma = torch.tensor([0.9])
    torch.testing.assert_close(ours.run(x, sigma, context), upstream(x, sigma, context), rtol=1e-4, atol=1e-5)


def test_gist_count():
    assert transformer.gist_count(10, 512) == 256
    assert transformer.gist_count(256, 512) == 256
    assert transformer.gist_count(1000, 512) == 384
    assert transformer.gist_count(2048, 512) == 512
    assert transformer.gist_count(9000, 512) == 512


@torch.no_grad()
def test_forward_reads_the_phase_of_the_step():
    _, ours = models()
    x, _ = inputs(batch=1)
    generator = torch.Generator().manual_seed(3)
    early, late = torch.randn(30, TEXT_DIM, generator=generator), torch.randn(42, TEXT_DIM, generator=generator)
    packed = phases.pack(late, early)[None]
    for sigma, text in ((0.95, early), (0.9, late), (0.3, late)):
        t = torch.tensor([sigma])
        torch.testing.assert_close(ours(x, t, packed, ref_latents=[]), ours.run(x, t, text[None]))


@torch.no_grad()
def test_forward_runs_samples_of_different_lengths_apart():
    _, ours = models()
    x, _ = inputs(batch=2)
    generator = torch.Generator().manual_seed(4)
    a = phases.pack(torch.randn(20, TEXT_DIM, generator=generator), torch.randn(25, TEXT_DIM, generator=generator))
    b = phases.pack(torch.randn(45, TEXT_DIM, generator=generator))
    # one batch of one width: a's 45 rows (20 late after 25 early) against b's 45
    context = torch.stack([a, b])
    t = torch.tensor([0.5, 0.5])
    out = ours(x, t, context, ref_latents=[])
    torch.testing.assert_close(out[:1], ours.run(x[:1], t[:1], a[25:, :-1][None]))
    torch.testing.assert_close(out[1:], ours.run(x[1:], t[1:], b[:, :-1][None]))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.no_grad()
def test_half_precision_close_to_float32(dtype):
    _, ours = models()
    x, context = inputs(batch=1)
    sigma = torch.tensor([0.6])
    expected = ours.run(x, sigma, context)
    out = ours.to(dtype).run(x.to(dtype), sigma, context.to(dtype))
    assert out.dtype == dtype
    assert torch.isfinite(out).all()
    assert (out.float() - expected).abs().mean() < 0.05 * expected.abs().mean()


@torch.no_grad()
def test_float16_survives_outputs_past_its_range():
    # an MLP output past 65504: ComfyUI's clamp would cut it, the scaled projection keeps it (and the norm after it
    # gives the same result as in float32)
    _, ours = models()
    with torch.no_grad():
        ours.layers[0].feed_forward.w2.weight.mul_(5000.0)
    x, context = inputs(batch=1)
    sigma = torch.tensor([0.6])
    expected = ours.run(x, sigma, context)
    out = ours.to(torch.float16).run(x.half(), sigma, context.half())
    assert torch.isfinite(out).all()
    assert (out.float() - expected).abs().mean() < 0.05 * expected.abs().mean()
