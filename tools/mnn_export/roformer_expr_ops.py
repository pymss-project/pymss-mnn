from __future__ import annotations

import json

import numpy as np
import torch

import MNN.expr as F


def const_tensor(tensor: torch.Tensor):
    array = tensor.detach().cpu().float().numpy()
    return F.const(np.ascontiguousarray(array), list(array.shape), F.NCHW, F.float)


def linear(x, module):
    y = F.matmul(x, const_tensor(module.weight), False, True)
    if module.bias is not None:
        y = y + const_tensor(module.bias)
    return y


def rms_norm(x, gamma, eps=1e-12):
    mean_sq = F.reduce_mean(x * x, [-1], True)
    return x * F.rsqrt(mean_sq + eps) * const_tensor(gamma)


def rotate_half(q, cos, sin):
    even = F.strided_slice(q, [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 2], 7, 15)
    odd = F.strided_slice(q, [0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 2], 7, 15)
    rotated_even = even * cos - odd * sin
    rotated_odd = odd * cos + even * sin
    return F.reshape(F.stack([rotated_even, rotated_odd], -1), q.shape)


def rotary_cos_sin(rotary_embed, seq_len: int):
    freqs = rotary_embed.freqs.detach().cpu().float().numpy()
    positions = np.arange(seq_len, dtype=np.float32)[:, None]
    angles = np.repeat(positions * freqs[None, :], 2, axis=-1)
    cos = np.ascontiguousarray(np.cos(angles)[None, :, None, ::2].astype(np.float32))
    sin = np.ascontiguousarray(np.sin(angles)[None, :, None, ::2].astype(np.float32))
    return F.const(cos, list(cos.shape), F.NCHW, F.float), F.const(sin, list(sin.shape), F.NCHW, F.float)


def mnn_attention(q, k, v, tokens: int):
    mask = F.const(np.zeros((tokens, tokens), dtype=np.float32), [tokens, tokens], F.NCHW, F.float)
    describe = json.dumps(
        {
            "type": "Attention",
            "main_type": "AttentionParam",
            "main": {"kv_cache": False},
        }
    )
    return F.jsonop([q, k, v, mask], describe, 1)[0]


def attention(attn, x, *, attention_op: str = "manual"):
    x_norm = rms_norm(x, attn.norm.gamma)
    qkv = linear(x_norm, attn.to_qkv)
    batch, tokens, _ = qkv.shape
    heads = int(attn.heads)
    head_dim = int(qkv.shape[-1] // (3 * heads))
    qkv = F.reshape(qkv, [batch, tokens, 3, heads, head_dim])
    q = F.squeeze(F.slice(qkv, [0, 0, 0, 0, 0], [batch, tokens, 1, heads, head_dim]), [2])
    k = F.squeeze(F.slice(qkv, [0, 0, 1, 0, 0], [batch, tokens, 1, heads, head_dim]), [2])
    v = F.squeeze(F.slice(qkv, [0, 0, 2, 0, 0], [batch, tokens, 1, heads, head_dim]), [2])
    if attn.rotary_embed is not None:
        cos, sin = rotary_cos_sin(attn.rotary_embed, tokens)
        q = rotate_half(q, cos, sin)
        k = rotate_half(k, cos, sin)

    if attention_op == "mnn":
        out = F.reshape(mnn_attention(q, k, v, tokens), [batch, tokens, heads, head_dim])
    elif attention_op == "manual":
        q = F.transpose(q, [0, 2, 1, 3])
        k = F.transpose(k, [0, 2, 1, 3])
        v = F.transpose(v, [0, 2, 1, 3])
        sim = F.matmul(q, k, False, True) * float(head_dim ** -0.5)
        out = F.matmul(F.softmax(sim, -1), v)
        out = F.transpose(out, [0, 2, 1, 3])
    else:
        raise ValueError(f"unsupported attention_op: {attention_op}")
    gates = F.sigmoid(linear(x_norm, attn.to_gates))
    out = out * F.unsqueeze(gates, -1)
    out = F.reshape(out, [batch, tokens, heads * head_dim])
    return linear(out, attn.to_out[0])


def feed_forward(ff, x):
    net = ff.net
    x = rms_norm(x, net[0].gamma)
    x = linear(x, net[1])
    x = x * 0.5 * (1.0 + F.erf(x / float(2.0 ** 0.5)))
    return linear(x, net[4])


def transformer(module, x, *, attention_op: str = "manual"):
    for attn, ff in module.layers:
        x = attention(attn, x, attention_op=attention_op) + x
        x = feed_forward(ff, x) + x
    if hasattr(module.norm, "gamma"):
        x = rms_norm(x, module.norm.gamma)
    return x


def transformer_attention_block(module, x, *, attention_op: str = "manual"):
    attn, _ = module.layers[0]
    return attention(attn, x, attention_op=attention_op) + x


def transformer_ffn_block(module, x):
    _, ff = module.layers[0]
    x = feed_forward(ff, x) + x
    if hasattr(module.norm, "gamma"):
        x = rms_norm(x, module.norm.gamma)
    return x


def band_split(model, stft_repr):
    batch, freq_channels, frames, complex_dim = stft_repr.shape
    flat = F.reshape(F.transpose(stft_repr, [0, 2, 1, 3]), [batch, frames, freq_channels * complex_dim])
    outs = []
    offset = 0
    for dim_in, feature in zip(model.band_split.dim_inputs, model.band_split.to_features):
        part = F.slice(flat, [0, 0, offset], [batch, frames, dim_in])
        part = rms_norm(part, feature[0].gamma)
        outs.append(linear(part, feature[1]))
        offset += dim_in
    return F.stack(outs, 2)


def mask_estimator_band(estimator, x, band_index: int):
    mlp_with_glu = estimator.to_freqs[band_index]
    band = x
    for layer in mlp_with_glu[0]:
        if isinstance(layer, torch.nn.Linear):
            band = linear(band, layer)
        elif isinstance(layer, torch.nn.Tanh):
            band = F.tanh(band)
        else:
            raise TypeError(f"unsupported mask MLP layer: {type(layer).__name__}")
    half = band.shape[-1] // 2
    value = F.slice(band, [0, 0, 0], [x.shape[0], x.shape[1], half])
    gate = F.slice(band, [0, 0, half], [x.shape[0], x.shape[1], half])
    return value * F.sigmoid(gate)


def mask_band(model, band_index: int, x):
    if hasattr(model.final_norm, "gamma"):
        x = rms_norm(x, model.final_norm.gamma)
    masks = [mask_estimator_band(estimator, x, band_index) for estimator in model.mask_estimators]
    return F.stack(masks, 1)
