from __future__ import annotations

from contextlib import contextmanager
import json

import numpy as np
import torch

import MNN.expr as F


_CONST_CACHE = None


@contextmanager
def const_cache():
    global _CONST_CACHE
    previous = _CONST_CACHE
    _CONST_CACHE = {}
    try:
        yield
    finally:
        _CONST_CACHE = previous


def const_tensor(tensor: torch.Tensor):
    if _CONST_CACHE is not None:
        key = (tensor.data_ptr(), tuple(tensor.shape))
        cached = _CONST_CACHE.get(key)
        if cached is not None:
            return cached
    array = tensor.detach().cpu().float().numpy()
    var = F.const(np.ascontiguousarray(array), list(array.shape), F.NCHW, F.float)
    if _CONST_CACHE is not None:
        _CONST_CACHE[key] = var
    return var


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


def rotary_cos_sin(rotary_embed, seq_len: int, batch: int = 1):
    freqs = rotary_embed.freqs.detach().cpu().float().numpy()
    positions = np.arange(seq_len, dtype=np.float32)[:, None]
    angles = np.repeat(positions * freqs[None, :], 2, axis=-1)
    cos = np.ascontiguousarray(np.cos(angles)[None, :, None, ::2].astype(np.float32))
    sin = np.ascontiguousarray(np.sin(angles)[None, :, None, ::2].astype(np.float32))
    if int(batch) != 1:
        cos = np.ascontiguousarray(np.repeat(cos, int(batch), axis=0))
        sin = np.ascontiguousarray(np.repeat(sin, int(batch), axis=0))
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
        cos, sin = rotary_cos_sin(attn.rotary_embed, tokens, batch=batch)
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
    x = F.gelu(x)
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


def roformer_block(model, start_layer: int, end_layer: int, x, *, attention_op: str = "manual"):
    for layer_index in range(start_layer, end_layer):
        time_transformer, freq_transformer = model.layers[layer_index]
        batch, frames, bands, dim = x.shape
        x = F.reshape(F.transpose(x, [0, 2, 1, 3]), [batch * bands, frames, dim])
        x = transformer(time_transformer, x, attention_op=attention_op)
        x = F.transpose(F.reshape(x, [batch, bands, frames, dim]), [0, 2, 1, 3])
        x = F.reshape(x, [batch * frames, bands, dim])
        x = transformer(freq_transformer, x, attention_op=attention_op)
        x = F.reshape(x, [batch, frames, bands, dim])
    return x


def roformer_block_grouped(model, start_layer: int, end_layer: int, x, *, time_group_size: int, attention_op: str = "manual"):
    if int(time_group_size) < 1:
        raise ValueError("time_group_size must be >= 1")

    for layer_index in range(start_layer, end_layer):
        time_transformer, freq_transformer = model.layers[layer_index]
        batch, frames, bands, dim = x.shape

        time_outputs = []
        for band_start in range(0, bands, int(time_group_size)):
            actual = min(int(time_group_size), bands - band_start)
            group = F.slice(x, [0, 0, band_start, 0], [batch, frames, actual, dim])
            group = F.reshape(F.transpose(group, [0, 2, 1, 3]), [batch * actual, frames, dim])
            group = transformer(time_transformer, group, attention_op=attention_op)
            group = F.transpose(F.reshape(group, [batch, actual, frames, dim]), [0, 2, 1, 3])
            time_outputs.append(group)
        x = F.concat(time_outputs, 2)

        x = F.reshape(x, [batch * frames, bands, dim])
        x = transformer(freq_transformer, x, attention_op=attention_op)
        x = F.reshape(x, [batch, frames, bands, dim])
    return x


def roformer_block_unrolled(model, start_layer: int, end_layer: int, x, *, freq_batch: int, attention_op: str = "manual"):
    if int(freq_batch) < 1:
        raise ValueError("freq_batch must be >= 1")

    for layer_index in range(start_layer, end_layer):
        time_transformer, freq_transformer = model.layers[layer_index]
        batch, frames, bands, dim = x.shape

        time_outputs = []
        for band_index in range(bands):
            band = F.squeeze(F.slice(x, [0, 0, band_index, 0], [batch, frames, 1, dim]), [2])
            band = transformer(time_transformer, band, attention_op=attention_op)
            time_outputs.append(F.unsqueeze(band, 2))
        x = F.concat(time_outputs, 2)

        freq_outputs = []
        for start in range(0, frames, int(freq_batch)):
            actual = min(int(freq_batch), frames - start)
            chunk = F.slice(x, [0, start, 0, 0], [batch, actual, bands, dim])
            chunk = F.reshape(chunk, [batch * actual, bands, dim])
            chunk = transformer(freq_transformer, chunk, attention_op=attention_op)
            freq_outputs.append(F.reshape(chunk, [batch, actual, bands, dim]))
        x = F.concat(freq_outputs, 1)
    return x


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


def mask_band_group(model, band_start: int, band_count: int, x):
    if hasattr(model.final_norm, "gamma"):
        gamma = model.final_norm.gamma
    else:
        gamma = None

    outputs = []
    batch, frames, _, dim = x.shape
    for local_index in range(band_count):
        band_index = band_start + local_index
        band = F.squeeze(F.slice(x, [0, 0, local_index, 0], [batch, frames, 1, dim]), [2])
        if gamma is not None:
            band = rms_norm(band, gamma)
        masks = [mask_estimator_band(estimator, band, band_index) for estimator in model.mask_estimators]
        outputs.append(F.stack(masks, 1))
    return F.concat(outputs, -1)
