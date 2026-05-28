import numpy as np
import torch
import torch.nn as nn

from .config import load_config


def get_model_from_config(model_type, config_path, model_kwargs_override=None):
    model_kwargs_override = model_kwargs_override or {}
    config = load_config(config_path)

    if model_type == 'mdx23c':
        from .modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        return TFC_TDF_net(config), config
    elif model_type == 'htdemucs':
        from .modules.demucs4ht import get_model
        return get_model(config), config
    elif model_type == 'mel_band_roformer':
        from .modules.bs_roformer import MelBandRoformer
        model_kwargs = dict(config.model)
        model_kwargs.update(model_kwargs_override)
        return MelBandRoformer(**model_kwargs), config
    elif model_type == 'bs_roformer':
        from .modules.bs_roformer import BSRoformer
        return BSRoformer(**dict(config.model)), config
    elif model_type == 'bs_roformer_hyperace':
        from .modules.bs_roformer import BSRoformerHyperACE
        return BSRoformerHyperACE(**dict(config.model)), config
    elif model_type == 'bandit':
        from .modules.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        return MultiMaskMultiSourceBandSplitRNNSimple(**config.model), config
    elif model_type == 'bandit_v2':
        from .modules.bandit_v2.bandit import Bandit
        return Bandit(**config.kwargs), config
    elif model_type == 'scnet':
        from .modules.scnet import SCNet
        return SCNet(**config.model), config
    elif model_type == 'apollo':
        from .modules.look2hear.apollo import Apollo
        return Apollo(**config.model), config
    elif model_type == 'vr':
        raise ValueError("VR models are loaded directly by MSSeparator and do not use YAML config loading")
    raise ValueError(f"Model type {model_type} not supported")

def _getWindowingArray(window_size, fade_size):
    if fade_size <= 0:
        return torch.ones(window_size)

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def _build_chunk_plan(total_length, chunk_size, step, fade_size):
    starts = list(range(0, total_length, step))
    normal_window = _getWindowingArray(chunk_size, fade_size)

    def window_for(start):
        length = min(chunk_size, total_length - start)
        if start != 0 and start + length < total_length:
            return normal_window
        window = normal_window.clone()
        if start == 0:
            window[:fade_size] = 1
        if start + length >= total_length:
            window[max(0, length - fade_size):length] = 1
        return window

    return starts, [window_for(start) for start in starts]


def _get_inference_step(config, chunk_size):
    overlap_size = int(config.inference.get('overlap_size', chunk_size // 2))
    if overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("inference.overlap_size must be >= 0 and < audio.chunk_size")
    return chunk_size - overlap_size


def _ensure_source_dim(x, chunk_batch):
    return x.unsqueeze(1) if x.ndim == chunk_batch.ndim else x


def _fit_tensor_length(x, length):
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return nn.functional.pad(x, (0, length - x.shape[-1]))
    return x


def _source_names(config):
    return config.training.instruments if config.training.target_instrument is None else [config.training.target_instrument]


def _source_count(config, source_indices=None):
    return len(_source_names(config)) if source_indices is None else len(source_indices)


def _prepare_mix_for_chunks(mix, border):
    length_init = mix.shape[-1]
    mix = mix.unsqueeze(0) if mix.ndim == 1 else mix
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode='reflect')
    return mix, length_init


def _init_overlap_buffers(config, mix, device, use_fast_path, source_indices=None):
    req_shape = (_source_count(config, source_indices),) + tuple(mix.shape)
    result_device = device if use_fast_path else 'cpu'
    counter_shape = (1, 1, mix.shape[1])
    result = torch.zeros(req_shape, dtype=torch.float32, device=result_device)
    counter = torch.zeros(counter_shape, dtype=torch.float32, device=result_device)
    return result, counter


def _select_sources(chunks, source_indices, already_selected=False):
    if source_indices is None or already_selected:
        return chunks
    index = torch.as_tensor(source_indices, device=chunks.device)
    return chunks.index_select(1, index)


def _run_model_chunk(model, arr, chunk_size, source_indices=None):
    target = model.module if isinstance(model, nn.DataParallel) else model
    chunks = _fit_tensor_length(_ensure_source_dim(model(arr), arr).float(), chunk_size)
    already_selected = (
        source_indices is not None
        and hasattr(target, "_active_source_indices")
        and chunks.shape[1] == len(source_indices)
    )
    return _select_sources(chunks, source_indices, already_selected=already_selected)


def _extract_chunk(mix, start, chunk_size):
    length = min(chunk_size, mix.shape[1] - start)
    part = mix[:, start:start + chunk_size]
    if length == chunk_size:
        return part, length
    if length > chunk_size // 2 + 1:
        part = nn.functional.pad(part, (0, chunk_size - length), mode='reflect')
    else:
        part = nn.functional.pad(part, (0, chunk_size - length, 0, 0), mode='constant', value=0)
    return part, length


def _finalize_overlap(result, counter, length_init, border):
    if length_init > 2 * border and border > 0:
        start, end = border, border + length_init
    else:
        start, end = 0, result.shape[-1]

    result = result[..., start:end]
    counter = counter[..., start:end]
    output_shape = result.shape[:-1] + (end - start,)

    if torch.device(result.device).type != "cuda":
        estimated_sources = (result / counter).cpu().numpy()
        np.nan_to_num(estimated_sources, copy=False, nan=0.0)
        return estimated_sources

    counter_min, counter_max = torch.aminmax(counter)
    divide_counter = bool((counter_min - 1).abs().item() > 1e-6 or (counter_max - 1).abs().item() > 1e-6)
    samples_per_chunk = max(1, (512 * 1024 * 1024) // (max(1, result.shape[0] * result.shape[1]) * 4))
    estimated_sources_t = torch.empty(output_shape, dtype=torch.float32, device="cpu")
    for offset in range(0, result.shape[-1], samples_per_chunk):
        chunk_end = min(offset + samples_per_chunk, result.shape[-1])
        source = result[..., offset:chunk_end]
        if divide_counter:
            source = source / counter[..., offset:chunk_end]
        estimated_sources_t[..., offset:chunk_end].copy_(source)
    estimated_sources = estimated_sources_t.numpy()
    if divide_counter:
        np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    return estimated_sources
