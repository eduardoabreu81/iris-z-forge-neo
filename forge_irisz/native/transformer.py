# Iris-Z support for Forge Neo
# The Iris-Z DiT: the FLUX.2 [klein] transformer with a modality embedding and one noise level per grid.
#
# Ported from speridlabs/iris-z-sdk (iris_z/model/model.py, Apache-2.0; see NOTICE), which adds both to Black
# Forest Labs' FLUX.2 transformer. The parameter names are the checkpoint's, so Forge's loader fills the model
# directly (quantized layers and LoRAs included); the layers are plain torch modules, built inside Forge's
# using_forge_operations, which swaps nn.Linear and nn.Embedding for its own.
#
# Every Iris-Z sequence holds two grids on the same positions: [RGB tokens, depth tokens], the depth grid at time
# coordinate 1. Each grid has its own noise level, and every per-token modulation (double-stream image stream,
# single stream, final layer) uses its grid's level; the text uses the level being sampled. A grid given as
# input sits at level 0 under its "given" modality. With no reference images the SDK's causal attention is plain
# full attention, which is what this runs.

import math

import torch
from torch import nn
from torch.nn import functional as F

from .. import contracts
from . import kernels

# modality ids (iris_z/rgbd.py)
RGB = 0
DEPTH = 1
GIVEN_RGB = 2
GIVEN_DEPTH = 3
DEPTH_TIME_COORDINATE = 1


def timestep_embedding(t, dim, max_period=10000, time_factor=1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def rope(pos, dim, theta):
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.float(), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def apply_rope(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def layer_norm(x):
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


class EmbedND(nn.Module):
    def __init__(self, theta, axes_dim):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids):
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(len(self.axes_dim))], dim=-3)
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    """The checkpoint's RMSNorm: a plain `scale` parameter, computed in float32."""

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rrms = torch.rsqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x.float() * rrms).to(x.dtype) * self.scale.to(device=x.device, dtype=x.dtype)


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, v):
        return self.query_norm(q).to(v), self.key_norm(k).to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim, bias=False)


class SiLUActivation(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim, double):
        super().__init__()
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=False)

    def forward(self, vec):
        """(shift, scale, gate[, shift2, scale2, gate2]), each [B, 1, D]."""
        return self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.img_attn = SelfAttention(hidden_size, num_heads)
        self.img_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden * 2, bias=False), SiLUActivation(),
                                     nn.Linear(mlp_hidden, hidden_size, bias=False))
        self.txt_attn = SelfAttention(hidden_size, num_heads)
        self.txt_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden * 2, bias=False), SiLUActivation(),
                                     nn.Linear(mlp_hidden, hidden_size, bias=False))


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=False)
        self.norm = QKNorm(hidden_size // num_heads)
        self.mlp_act = SiLUActivation()


class LastLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec)[:, None, :].chunk(2, dim=-1)
        return self.linear(torch.addcmul(shift, layer_norm(x), 1 + scale))


def _heads(qkv, num_heads):
    batch, length, _ = qkv.shape
    return qkv.view(batch, length, 3, num_heads, -1).permute(2, 0, 3, 1, 4)


def _modulate(x, spans, shifts, scales):
    """(1 + scale) * x + shift, with each span of tokens taking its own grid's modulation."""
    if len(spans) == 1:
        return torch.addcmul(shifts[0], x, 1 + scales[0])
    out = torch.empty_like(x)
    for (start, end), shift, scale in zip(spans, shifts, scales):
        out[:, start:end] = torch.addcmul(shift, x[:, start:end], 1 + scale)
    return out


def _gated_add(x, y, spans, gates):
    """x += gate * y in place, span by span."""
    for (start, end), gate in zip(spans, gates):
        x[:, start:end] += y[:, start:end] * gate
    return x


def _double_block(block, img, txt, pe, img_spans, img_mods, txt_mod, options):
    heads = block.num_heads
    x = _modulate(layer_norm(img), img_spans, [m[0] for m in img_mods], [m[1] for m in img_mods])
    img_q, img_k, img_v = _heads(block.img_attn.qkv(x), heads)
    img_q, img_k = block.img_attn.norm(img_q, img_k, img_v)
    del x

    t = torch.addcmul(txt_mod[0], layer_norm(txt), 1 + txt_mod[1])
    txt_q, txt_k, txt_v = _heads(block.txt_attn.qkv(t), heads)
    txt_q, txt_k = block.txt_attn.norm(txt_q, txt_k, txt_v)
    del t

    q, k = apply_rope(torch.cat((txt_q, img_q), dim=2), torch.cat((txt_k, img_k), dim=2), pe)
    v = torch.cat((txt_v, img_v), dim=2)
    del txt_q, txt_k, txt_v, img_q, img_k, img_v
    attn = kernels.attention(q, k, v, options)
    del q, k, v
    num_txt = txt.shape[1]
    txt_attn, img_attn = attn[:, :num_txt], attn[:, num_txt:]

    _gated_add(img, block.img_attn.proj(img_attn), img_spans, [m[2] for m in img_mods])
    y = _modulate(layer_norm(img), img_spans, [m[3] for m in img_mods], [m[4] for m in img_mods])
    _gated_add(img, block.img_mlp(y), img_spans, [m[5] for m in img_mods])
    del y

    txt = txt + txt_mod[2] * block.txt_attn.proj(txt_attn)
    txt = txt + txt_mod[5] * block.txt_mlp(torch.addcmul(txt_mod[3], layer_norm(txt), 1 + txt_mod[4]))
    return img, kernels.fp16_fix(txt)


def _single_block(block, x, pe, spans, mods, options):
    hidden = block.hidden_size
    y = _modulate(layer_norm(x), spans, [m[0] for m in mods], [m[1] for m in mods])
    qkv, mlp = torch.split(block.linear1(y), [3 * hidden, block.mlp_hidden_dim * 2], dim=-1)
    del y
    q, k, v = _heads(qkv, block.num_heads)
    del qkv
    q, k = block.norm(q, k, v)
    q, k = apply_rope(q, k, pe)
    attn = kernels.attention(q, k, v, options)
    del q, k, v
    out = block.linear2(torch.cat((attn, block.mlp_act(mlp)), dim=2))
    _gated_add(x, out, spans, [m[2] for m in mods])
    return kernels.fp16_fix(x)


def grid_ids(height, width, time_coordinate, device):
    rows = torch.arange(height, device=device, dtype=torch.float32)
    cols = torch.arange(width, device=device, dtype=torch.float32)
    ids = torch.zeros(height, width, 4, device=device, dtype=torch.float32)
    ids[..., 0] = time_coordinate
    ids[..., 1] = rows[:, None]
    ids[..., 2] = cols[None, :]
    return ids.reshape(height * width, 4)


def text_ids(length, device):
    ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
    return ids


class IrisZDiT(nn.Module):
    def __init__(self, in_channels=128, context_in_dim=12288, hidden_size=4096, num_heads=32, depth=8,
                 depth_single_blocks=24, axes_dim=(32, 32, 32, 32), theta=2000, mlp_ratio=3.0, num_modalities=4,
                 **ignored):
        super().__init__()
        if hidden_size % num_heads or sum(axes_dim) != hidden_size // num_heads:
            raise ValueError(f"Iris-Z: hidden size {hidden_size} does not fit {num_heads} heads of {list(axes_dim)}")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = EmbedND(theta, axes_dim)
        self.img_in = nn.Linear(in_channels, hidden_size, bias=False)
        self.time_in = MLPEmbedder(256, hidden_size)
        self.txt_in = nn.Linear(context_in_dim, hidden_size, bias=False)
        self.double_blocks = nn.ModuleList([DoubleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth_single_blocks)])
        self.double_stream_modulation_img = Modulation(hidden_size, double=True)
        self.double_stream_modulation_txt = Modulation(hidden_size, double=True)
        self.single_stream_modulation = Modulation(hidden_size, double=False)
        self.final_layer = LastLayer(hidden_size, in_channels)
        self.modality_in = nn.Embedding(num_modalities, hidden_size)

    def forward(self, x, timestep, context, transformer_options=None, **kwargs):
        """Forge's call: x is the grid being sampled [B, C, h, w], timestep its sigma [B]. The other grid comes from
        the generation's plan (contracts.py): pure noise next to an image, the given image next to its depth, or the
        given depth next to the image painted over it."""
        sigma = timestep.float()
        plan = contracts.current_plan()
        if plan is not None and plan.mode == contracts.DEPTH:
            given = contracts.given_rgb(plan, x)
            _, depth = self.forward_grids(given, x, torch.zeros_like(sigma), sigma, sigma, (GIVEN_RGB, DEPTH),
                                          context, transformer_options)
            return depth
        if plan is not None and plan.mode == contracts.PAINT:
            given = contracts.given_depth(plan, x)
            rgb, _ = self.forward_grids(x, given, sigma, torch.zeros_like(sigma), sigma, (RGB, GIVEN_DEPTH),
                                        context, transformer_options)
            return rgb
        noise = contracts.companion_noise(plan, x)
        rgb, _ = self.forward_grids(x, noise, sigma, torch.ones_like(sigma), sigma, (RGB, DEPTH), context,
                                    transformer_options)
        return rgb

    def forward_grids(self, rgb, depth, rgb_t, depth_t, main_t, modalities, context, transformer_options=None):
        """Velocity of both grids [B, C, h, w]. rgb_t / depth_t are each grid's noise level, main_t the level the
        text is modulated with (the grid being sampled); modalities are the (rgb, depth) modality ids."""
        options = transformer_options or {}
        batch, _, height, width = rgb.shape
        n = height * width
        dtype = rgb.dtype
        device = rgb.device

        tokens = torch.cat((rgb.flatten(2).transpose(1, 2), depth.flatten(2).transpose(1, 2)), dim=1)
        ids = torch.cat((text_ids(context.shape[1], device), grid_ids(height, width, 0, device),
                         grid_ids(height, width, DEPTH_TIME_COORDINATE, device)))
        pe = self.pe_embedder(ids[None])

        def embed(t):
            return self.time_in(timestep_embedding(t, 256).to(device=device, dtype=dtype))

        vec, rgb_vec, depth_vec = embed(main_t), embed(rgb_t), embed(depth_t)

        img = self.img_in(tokens)
        modality = self.modality_in(torch.tensor(modalities, device=device)).to(img.dtype)
        img[:, :n] += modality[0]
        img[:, n:] += modality[1]
        txt = self.txt_in(context.to(dtype))
        del tokens

        txt_mod = self.double_stream_modulation_txt(vec)
        img_spans = ((0, n), (n, 2 * n))
        img_mods = (self.double_stream_modulation_img(rgb_vec), self.double_stream_modulation_img(depth_vec))
        for block in self.double_blocks:
            img, txt = _double_block(block, img, txt, pe, img_spans, img_mods, txt_mod, options)

        num_txt = txt.shape[1]
        x = kernels.fp16_fix(torch.cat((txt, img), dim=1))
        del txt, img
        spans = ((0, num_txt), (num_txt, num_txt + n), (num_txt + n, num_txt + 2 * n))
        mods = tuple(self.single_stream_modulation(v) for v in (vec, rgb_vec, depth_vec))
        for block in self.single_blocks:
            x = _single_block(block, x, pe, spans, mods, options)

        x = x[:, num_txt:]
        rgb_out = self.final_layer(x[:, :n], rgb_vec)
        depth_out = self.final_layer(x[:, n:], depth_vec)

        def grid(out):
            return out.transpose(1, 2).reshape(batch, self.out_channels, height, width)

        return grid(rgb_out), grid(depth_out)
