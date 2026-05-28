#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymss.audio_io import load_audio, save_audio  # noqa: E402
from pymss.config import load_config  # noqa: E402
from pymss.modules.bs_roformer.common import istft_roformer, stft_roformer  # noqa: E402
from pymss.separator import _model_is_stereo, _prepare_mix_channels  # noqa: E402
from pymss.utils import (  # noqa: E402
    _build_chunk_plan,
    _extract_chunk,
    _finalize_overlap,
    _get_inference_step,
    _init_overlap_buffers,
    _run_model_chunk,
    _source_names,
)
from tools.mnn_export.presets import default_out_dir, get_preset, preset_names  # noqa: E402
from tools.mnn_export.roformer_mnn import build_separator, metrics, prepare_model_for_export  # noqa: E402
from tools.mnn_export.validate_expr_micro_segments import ExprMicroRuntime  # noqa: E402
from tools.mnn_export.roformer_mnn import MaskHeadWrapper  # noqa: E402


class MNNMaskCoreRoformer(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module, *, backend: str, threads: int, segment_dir: Path, time_batch: int = 1, freq_batch: int = 16):
        super().__init__()
        self.torch_model = prepare_model_for_export(torch_model.cpu()).eval()
        self.backend = backend
        self.threads = threads
        self.segment_runtime = ExprMicroRuntime(
            segment_dir,
            depth=len(self.torch_model.layers),
            time_batch=time_batch,
            freq_batch=freq_batch,
            backend=backend,
            threads=threads,
        )

    def forward(self, raw_audio):
        stft_repr, context = stft_roformer(self.torch_model, raw_audio)
        mask_input = stft_repr
        if self.torch_model.__class__.__name__ == "MelBandRoformer":
            batch_indices = torch.arange(context.batch, device=stft_repr.device)[..., None]
            mask_input = stft_repr[batch_indices, self.torch_model.freq_indices.to(stft_repr.device)]
        stft_np = mask_input.detach().cpu().numpy()
        mask_np = self.segment_runtime(stft_np)
        mask = torch.from_numpy(mask_np).to(dtype=stft_repr.dtype)
        if self.torch_model.__class__.__name__ == "MelBandRoformer":
            masked = self._mask_mbr(stft_repr, mask, context)
            length = context.audio_length if self.torch_model.match_input_audio_length else None
        else:
            stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1))
            masked = stft_complex * torch.view_as_complex(mask).type(stft_complex.dtype)
            length = context.audio_length
        return istft_roformer(self.torch_model, masked, context, length)

    def _mask_mbr(self, stft_repr, mask, context):
        stft_complex = torch.view_as_complex(stft_repr.unsqueeze(1))
        masks = torch.view_as_complex(mask.contiguous()).to(dtype=stft_complex.dtype)
        freq_indices = self.torch_model.freq_indices.cpu()
        scatter_indices = freq_indices[None, None, :, None].expand(
            context.batch,
            self.torch_model.num_stems,
            -1,
            stft_complex.shape[-1],
        )
        masks_summed = stft_complex.new_zeros(context.batch, self.torch_model.num_stems, stft_complex.shape[2], stft_complex.shape[-1])
        masks_summed.scatter_add_(2, scatter_indices, masks)
        denom = self.torch_model.num_bands_per_channel_freq.cpu().clamp(min=1e-8)
        return stft_complex * (masks_summed / denom)


class _NullLogger:
    def warning(self, *_args, **_kwargs):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Run end-to-end RoFormer separation with an MNN mask-core model.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {preset_names()}")
    parser.add_argument("--audio", type=Path, default=ROOT / "test.m4a")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument("--mnn-backend", default="CPU")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=None, help="Optional short validation duration.")
    parser.add_argument("--time-batch", type=int, default=1)
    parser.add_argument("--freq-batch", type=int, default=16)
    parser.add_argument("--torch-mask-head", action="store_true", help="Fallback: run the final mask head in PyTorch.")
    return parser.parse_args()


def source_dict(config, estimated_sources):
    names = _source_names(config)
    return {name: value for name, value in zip(names, estimated_sources)}


def demix_with_model(config, model, mix):
    chunk_size = int(config.audio.chunk_size)
    step = _get_inference_step(config, chunk_size)
    border = chunk_size - step
    fade_size = min(chunk_size // 10, border)
    batch_size = int(config.inference.batch_size)
    mix_t = torch.tensor(mix, dtype=torch.float32)
    mix_padded, length_init = __import__("pymss.utils", fromlist=["_prepare_mix_for_chunks"])._prepare_mix_for_chunks(mix_t, border)
    starts, windows = _build_chunk_plan(mix_padded.shape[1], chunk_size, step, fade_size)
    result, counter = _init_overlap_buffers(config, mix_padded, "cpu", False)
    with torch.inference_mode():
        for batch_start in range(0, len(starts), batch_size):
            batch_indices = range(batch_start, min(batch_start + batch_size, len(starts)))
            batch = [(_extract_chunk(mix_padded, starts[index], chunk_size), index) for index in batch_indices]
            chunks = _run_model_chunk(model, torch.stack([chunk for (chunk, _), _ in batch], dim=0), chunk_size)
            for j, ((_, length), index) in enumerate(batch):
                start = starts[index]
                window = windows[index].to(dtype=torch.float32)[:length]
                result[..., start : start + length] += chunks[j, ..., :length].float() * window
                counter[..., start : start + length] += window
    return source_dict(config, _finalize_overlap(result, counter, length_init, border))


def save_outputs(store_dir: Path, outputs: dict[str, np.ndarray], sample_rate: int, stem: str) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    params = {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k", "m4a_bit_rate": "192k", "m4a_aac_at_quality": 2}
    for name, audio in outputs.items():
        save_audio(str(store_dir / f"{stem}_{name}.wav"), np.asarray(audio).T, sample_rate, "wav", params)


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    segment_dir = args.out_dir / preset.name / "expr_micro_segments"
    if not segment_dir.exists():
        raise FileNotFoundError(f"missing MNN micro segment directory: {segment_dir}")

    separator = build_separator(preset, batch_size=1, device="cpu")
    separator.config.inference.batch_size = 1
    config = separator.config
    cfg_for_audio = load_config(preset.config_path)
    duration = args.seconds if args.seconds and args.seconds > 0 else None
    mix, sr = load_audio(str(args.audio), sr=int(config.audio.sample_rate), mono=False, duration=duration)
    mix = _prepare_mix_channels(np.asarray(mix, dtype=np.float32), _model_is_stereo(preset.model_type, cfg_for_audio), _NullLogger())

    mnn_model = MNNMaskCoreRoformer(
        separator.model,
        backend=args.mnn_backend,
        threads=args.threads,
        segment_dir=segment_dir,
        time_batch=args.time_batch,
        freq_batch=args.freq_batch,
    )
    if args.torch_mask_head and mnn_model.segment_runtime is not None:
        mnn_model.segment_runtime.mask_head = MaskHeadWrapper(separator.model).eval()
    started = time.perf_counter()
    mnn_outputs = demix_with_model(config, mnn_model, mix)
    mnn_seconds = time.perf_counter() - started

    result = {
        "preset": preset.name,
        "audio": str(args.audio),
        "sample_rate": sr,
        "duration_seconds": float(mix.shape[-1] / sr),
        "mask_head": mnn_model.segment_runtime.mask_head_kind,
        "mnn_seconds": mnn_seconds,
        "outputs": {name: list(value.shape) for name, value in mnn_outputs.items()},
    }
    if args.compare_torch:
        started = time.perf_counter()
        torch_outputs = demix_with_model(config, separator.model, mix)
        torch_seconds = time.perf_counter() - started
        result["torch_seconds"] = torch_seconds
        result["mnn_vs_torch"] = {
            name: metrics(torch_outputs[name], mnn_outputs[name])
            for name in mnn_outputs
            if name in torch_outputs
        }

    store_dir = args.store_dir or (args.out_dir / preset.name / "audio_outputs")
    save_outputs(store_dir, mnn_outputs, sr, args.audio.stem)
    result["store_dir"] = str(store_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    separator.del_cache()


if __name__ == "__main__":
    main()
