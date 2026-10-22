# Transcription of ComfyUI's Fibo 2.0 DiT, for the tests only: comfy/ldm/fibo2/model.py and the pieces it imports
# (comfy/ldm/lumina/model.py: JointAttention, FinalLayer, modulate, apply_gate, clamp_fp16, pos_ids_x;
# comfy/ldm/flux/layers.py and math.py: EmbedND, rope, apply_rope; comfy/ldm/modules/diffusionmodules: the
# TimestepEmbedder and its timestep_embedding), Comfy-Org/ComfyUI commit 87ed5a066d9c4660ef42100fd01a888da6994856,
# GNU General Public License v3.0.
#
# ComfyUI's operations, attention dispatch, prefetching and wrappers are replaced with plain torch: Linear,
# RMSNorm, LayerNorm and scaled_dot_product_attention. The training path of JointAttention (norm, then rope) stands
# for its fused rms_rope kernel. Everything else is as upstream wrote it.

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope(pos, dim, theta):
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=pos.device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float64), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope1(x, freqs_cis):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


def invert_slices(slices, length):
    sorted_slices = sorted(slices)
    result = []
    current = 0
    for start, end in sorted_slices:
        if current < start:
            result.append((current, start))
        current = max(current, end)
    if current < length:
        result.append((current, length))
    return result


def modulate(x, scale, timestep_zero_index=None):
    if timestep_zero_index is None:
        return x * (1 + scale.unsqueeze(1))
    else:
        scale = (1 + scale.unsqueeze(1))
        actual_batch = scale.size(0) // 2
        slices = timestep_zero_index
        invert = invert_slices(timestep_zero_index, x.shape[1])
        for s in slices:
            x[:, s[0]:s[1]] *= scale[actual_batch:]
        for s in invert:
            x[:, s[0]:s[1]] *= scale[:actual_batch]
        return x


def apply_gate(gate, x, timestep_zero_index=None):
    if timestep_zero_index is None:
        return gate * x
    else:
        actual_batch = gate.size(0) // 2
        slices = timestep_zero_index
        invert = invert_slices(timestep_zero_index, x.shape[1])
        for s in slices:
            x[:, s[0]:s[1]] *= gate[actual_batch:]
        for s in invert:
            x[:, s[0]:s[1]] *= gate[:actual_batch]
        return x


def clamp_fp16(x):
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


def pos_ids_x(start_t, H_tokens, W_tokens, batch_size, device):
    x_pos_ids = torch.zeros((batch_size, H_tokens * W_tokens, 3), dtype=torch.float32, device=device)
    x_pos_ids[:, :, 0] = start_t
    x_pos_ids[:, :, 1] = torch.arange(H_tokens, dtype=torch.float32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
    x_pos_ids[:, :, 2] = torch.arange(W_tokens, dtype=torch.float32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
    return x_pos_ids


def sdpa(q, k, v):
    # (B, H, N, D) -> (B, N, H * D)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).flatten(2)


class JointAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_local_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * n_heads * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, x_mask, freqs_cis):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = torch.split(self.qkv(x), [self.n_local_heads * self.head_dim] * 3, dim=-1)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xq = apply_rope1(self.q_norm(xq), freqs_cis)
        xk = apply_rope1(self.k_norm(xk), freqs_cis)
        return self.out(sdpa(xq.movedim(1, 2), xk.movedim(1, 2), xv.movedim(1, 2)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(min(hidden_size, 256), hidden_size, bias=True))

    def forward(self, x, c, timestep_zero_index=None):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale, timestep_zero_index=timestep_zero_index)
        return self.linear(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, output_size=None, max_period=10000):
        super().__init__()
        if output_size is None:
            output_size = hidden_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size, bias=True), nn.SiLU(), nn.Linear(hidden_size, output_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def forward(self, t, dtype, **kwargs):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, max_period=self.max_period).to(dtype)
        return self.mlp(t_freq)


# comfy/ldm/fibo2/model.py


def residual(x, y, gate=None):
    return x.add_(y) if gate is None else x.addcmul_(y, gate)


def gist_count(text_len, max_gist):
    ratio = min(1.0, max(0.0, (text_len - 256) / (2048 - 256)))
    return max(256, min(max_gist, math.ceil((256 + ratio * (max_gist - 256)) / 64) * 64))


class Attention(nn.Module):
    def __init__(self, dim, n_heads, kv_dim, norm_eps):
        super().__init__()
        self.n_heads = n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, 2 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.norm_q = nn.RMSNorm(dim // n_heads, eps=norm_eps)
        self.norm_k = nn.RMSNorm(dim // n_heads, eps=norm_eps)

    def forward(self, x, context):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.n_heads, -1)))
        k, v = self.to_kv(context).unflatten(-1, (2, self.n_heads, -1)).unbind(2)
        k = self.norm_k(k)
        return self.to_out(sdpa(q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, fused):
        super().__init__()
        self.fused = fused
        if fused:
            self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        if self.fused:
            gate, up = self.w13(x).chunk(2, dim=-1)
            return self.w2(F.silu(gate) * up)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, norm_eps, cap_feat_dim=None):
        super().__init__()
        self.attention = JointAttention(dim, n_heads)
        self.attention.q_norm.eps = self.attention.k_norm.eps = norm_eps
        self.feed_forward = FeedForward(dim, ffn_dim, fused_ffn)
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.Linear(256, 4 * dim, bias=True))
        self.cross_attention = None
        if cap_feat_dim is not None:
            self.cross_attn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
            self.cross_attention = Attention(dim, n_heads, cap_feat_dim, norm_eps)
            self.cross_attn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = nn.Parameter(torch.empty(1))

    def forward(self, x, freqs_cis, adaln_input, timestep_zero_index=None, context=None):
        scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
        y = self.attention(modulate(self.attention_norm1(x), scale_msa, timestep_zero_index=timestep_zero_index), None, freqs_cis)
        x = residual(x, apply_gate(gate_msa.unsqueeze(1).tanh(), self.attention_norm2(clamp_fp16(y)), timestep_zero_index=timestep_zero_index))
        if self.cross_attention is not None:
            y = self.cross_attention(self.cross_attn_norm1(x), context)
            x = residual(x, self.cross_attn_norm2(y), self.cross_attn_gate.to(y).tanh())
        y = self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp, timestep_zero_index=timestep_zero_index))
        return residual(x, apply_gate(gate_mlp.unsqueeze(1).tanh(), self.ffn_norm2(clamp_fp16(y)), timestep_zero_index=timestep_zero_index))


class PerceiverLayer(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, norm_eps):
        super().__init__()
        self.cross_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attention = Attention(dim, n_heads, cap_feat_dim, norm_eps)
        self.self_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.self_attention = Attention(dim, n_heads, dim, norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim, ffn_dim, fused_ffn)

    def forward(self, x, context):
        x = x + self.cross_attention(self.cross_attn_norm(x), context)
        y = self.self_attn_norm(x)
        x = x + self.self_attention(y, y)
        return x + self.ffn(self.ffn_norm(x))


class Perceiver(nn.Module):
    def __init__(self, dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, num_queries, num_layers, norm_eps):
        super().__init__()
        self.latent_queries = nn.Parameter(torch.empty(num_queries, dim))
        self.layers = nn.ModuleList([PerceiverLayer(dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, norm_eps) for _ in range(num_layers)])
        self.norm_out = nn.RMSNorm(dim, eps=norm_eps)

    def forward(self, context, num_gist):
        x = self.latent_queries[:num_gist].to(context).unsqueeze(0).expand(context.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, context)
        return self.norm_out(x)


class Fibo2(nn.Module):
    def __init__(self, in_channels=128, dim=3584, n_heads=28, n_layers=44, n_refiner_layers=2, injection_layers=(6, 14, 22, 30, 38), cap_feat_dim=7680,
                 perceiver_layers=2, max_gist_tokens=512, ffn_dim=9728, fused_ffn=True, norm_eps=1e-5, image_model=None):
        super().__init__()
        self.max_gist_tokens = max_gist_tokens
        self.injection_layers = injection_layers

        self.x_embedder = nn.Linear(in_channels, dim, bias=True)
        self.t_embedder = TimestepEmbedder(1024, output_size=256)
        self.noise_refiner = nn.ModuleList([Block(dim, n_heads, ffn_dim, fused_ffn, norm_eps) for _ in range(n_refiner_layers)])
        self.perceiver = Perceiver(dim, n_heads, ffn_dim, fused_ffn, cap_feat_dim, max_gist_tokens, perceiver_layers, norm_eps)
        self.layers = nn.ModuleList([Block(dim, n_heads, ffn_dim, fused_ffn, norm_eps, cap_feat_dim if i in injection_layers else None) for i in range(n_layers)])
        self.final_layer = FinalLayer(dim, 1, in_channels)
        self.rope_embedder = EmbedND(dim=dim // n_heads, theta=256.0, axes_dim=[32, 48, 48])

    def forward(self, x, timestep, context, ref_latents=[]):
        B, C, H, W = x.shape
        dtype, device = x.dtype, x.device

        adaln_mixed = self.t_embedder(torch.cat((torch.zeros_like(timestep), timestep * 1000)), dtype)
        adaln_noisy = adaln_mixed[B:]

        num_gist = gist_count(context.shape[1], self.max_gist_tokens)
        plane = self.max_gist_tokens + 1
        tokens = [x.flatten(2).transpose(1, 2)]
        ids = [torch.zeros(1, num_gist, 3, dtype=torch.float32, device=device), pos_ids_x(plane, H, W, 1, device)]
        ids[0][0, :, 0] = torch.arange(1, num_gist + 1, dtype=torch.float32, device=device)
        for k, ref in enumerate(ref_latents):
            tokens.append(ref.repeat(B, 1, 1, 1).flatten(2).transpose(1, 2))
            ids.append(pos_ids_x(plane + 1 + k, ref.shape[-2], ref.shape[-1], 1, device))
        img = self.x_embedder(torch.cat(tokens, dim=1))
        freqs_cis = self.rope_embedder(torch.cat(ids, dim=1)).movedim(1, 2)
        del tokens

        n_img = H * W
        refiner_adaln, refiner_index = (adaln_mixed, [(0, n_img)]) if len(ref_latents) > 0 else (adaln_noisy, None)
        for block in self.noise_refiner:
            img = block(img, freqs_cis[:, num_gist:], refiner_adaln, refiner_index)

        bundles = context.chunk(1 + len(self.injection_layers), dim=-1)
        gist = self.perceiver(bundles[0], num_gist)
        text = dict(zip(self.injection_layers, bundles[1:]))

        x = torch.cat((gist, img), dim=1)
        image_index = [(num_gist, num_gist + n_img)]
        for i, block in enumerate(self.layers):
            x = block(x, freqs_cis, adaln_mixed, image_index, text.get(i))

        out = self.final_layer(x[:, num_gist:num_gist + n_img], adaln_noisy)
        return out.transpose(1, 2).reshape(B, C, H, W)
