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
from tools.mnn_export.non_roformer_mnn import (  # noqa: E402
    apollo_core_input,
    apollo_finish_output,
    bandit_core_input,
    bandit_finish_output,
    build_non_roformer_separator,
    default_out_dir,
    get_non_roformer_preset,
    htdemucs_core_inputs,
    htdemucs_finish_output,
    mdx23c_core_input,
    mdx23c_finish_output,
    metrics,
    non_roformer_preset_names,
    scnet_istft_output,
    scnet_stft_input,
)
from tools.mnn_export.roformer_mnn import MNNMultiRunner, MNNRunner  # noqa: E402


class _NullLogger:
    def warning(self, *_args, **_kwargs):
        pass


class MNNCoreSCNet(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module, mnn_path: Path, *, backend: str, threads: int):
        super().__init__()
        self.torch_model = torch_model.cpu().float().eval()
        self.runner = MNNRunner(mnn_path, input_name="stft_repr", output_name="stft_out", backend=backend, threads=threads)

    def forward(self, raw_audio):
        stft_repr, padding = scnet_stft_input(self.torch_model, raw_audio)
        stft_out_np = self.runner(stft_repr.detach().cpu().numpy())
        stft_out = torch.from_numpy(stft_out_np).to(dtype=raw_audio.dtype)
        return scnet_istft_output(self.torch_model, stft_out, padding=padding)


class MNNCoreHTDemucs(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module, mnn_path: Path, *, backend: str, threads: int):
        super().__init__()
        self.torch_model = torch_model.cpu().float().eval()
        self.runner = MNNMultiRunner(
            mnn_path,
            input_names=("mag", "mix"),
            output_names=("mask", "time"),
            backend=backend,
            threads=threads,
        )

    def forward(self, raw_audio):
        mag, mix, z = htdemucs_core_inputs(self.torch_model, raw_audio)
        outputs = self.runner(
            {
                "mag": mag.detach().cpu().numpy(),
                "mix": mix.detach().cpu().numpy(),
            }
        )
        mask = torch.from_numpy(outputs["mask"]).to(dtype=raw_audio.dtype)
        time = torch.from_numpy(outputs["time"]).to(dtype=raw_audio.dtype)
        return htdemucs_finish_output(self.torch_model, z, mask, time, length=raw_audio.shape[-1])


class MNNCoreMDX23C(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module, mnn_path: Path, *, backend: str, threads: int):
        super().__init__()
        self.torch_model = torch_model.cpu().float().eval()
        self.runner = MNNRunner(mnn_path, input_name="cws_spec", output_name="cac_spec", backend=backend, threads=threads)

    def forward(self, raw_audio):
        cws_spec = mdx23c_core_input(self.torch_model, raw_audio)
        cac_spec_np = self.runner(cws_spec.detach().cpu().numpy())
        cac_spec = torch.from_numpy(cac_spec_np).to(dtype=raw_audio.dtype)
        return mdx23c_finish_output(self.torch_model, cac_spec)


class MNNCoreApollo(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module, mnn_path: Path, *, backend: str, threads: int):
        super().__init__()
        self.torch_model = torch_model.cpu().float().eval()
        self.runner = MNNRunner(mnn_path, input_name="stft_ri", output_name="est_spec_ri", backend=backend, threads=threads)

    def forward(self, raw_audio):
        batch, channels, length = raw_audio.shape
        stft_ri = apollo_core_input(self.torch_model, raw_audio)
        est_spec_np = self.runner(stft_ri.detach().cpu().numpy())
        est_spec_ri = torch.from_numpy(est_spec_np).to(dtype=raw_audio.dtype)
        return apollo_finish_output(self.torch_model, est_spec_ri, batch=batch, channels=channels, length=length)


class MNNCoreBandit(torch.nn.Module):
    def __init__(self, torch_model: torch.nn.Module, mnn_path: Path, *, backend: str, threads: int, flatten_channels: bool = False):
        super().__init__()
        self.torch_model = torch_model.cpu().float().eval()
        self.flatten_channels = flatten_channels
        self.runner = MNNRunner(mnn_path, input_name="stft_ri", output_name="est_spec_ri", backend=backend, threads=threads)

    def forward(self, raw_audio):
        batch, channels, length = raw_audio.shape
        stft_ri = bandit_core_input(self.torch_model, raw_audio, flatten_channels=self.flatten_channels)
        est_spec_np = self.runner(stft_ri.detach().cpu().numpy())
        est_spec_ri = torch.from_numpy(est_spec_np).to(dtype=raw_audio.dtype)
        return bandit_finish_output(
            self.torch_model,
            est_spec_ri,
            length=length,
            batch=batch if self.flatten_channels else None,
            channels=channels if self.flatten_channels else None,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end non-RoFormer separation with an MNN neural core.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {non_roformer_preset_names()}")
    parser.add_argument("--audio", type=Path, default=ROOT / "test.m4a")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument("--mnn-backend", default="CPU")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=None)
    return parser.parse_args()


def source_dict(config, estimated_sources):
    names = _source_names(config)
    return {name: value for name, value in zip(names, estimated_sources)}


def sample_rate_from_config(config) -> int:
    return int(config.audio.get("sample_rate", config.training.get("samplerate", 44100)))


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
    preset = get_non_roformer_preset(args.preset)
    mnn_path = args.out_dir / preset.name / f"{preset.name}_core.mnn"
    if not mnn_path.exists():
        raise FileNotFoundError(f"missing MNN model: {mnn_path}")

    separator = build_non_roformer_separator(preset, batch_size=1, device="cpu")
    config = separator.config
    config.inference.batch_size = 1
    config.inference.overlap_size = int(preset.overlap_size)
    config.audio.chunk_size = int(preset.chunk_size)

    duration = args.seconds if args.seconds and args.seconds > 0 else None
    mix, sr = load_audio(str(args.audio), sr=sample_rate_from_config(config), mono=False, duration=duration)
    cfg_for_audio = load_config(preset.config_path)
    mix = _prepare_mix_channels(np.asarray(mix, dtype=np.float32), _model_is_stereo(preset.model_type, cfg_for_audio), _NullLogger())

    if preset.core_kind == "scnet_core":
        mnn_model = MNNCoreSCNet(separator.model, mnn_path, backend=args.mnn_backend, threads=args.threads)
    elif preset.core_kind == "htdemucs_core":
        mnn_model = MNNCoreHTDemucs(separator.model, mnn_path, backend=args.mnn_backend, threads=args.threads)
    elif preset.core_kind == "mdx23c_core":
        mnn_model = MNNCoreMDX23C(separator.model, mnn_path, backend=args.mnn_backend, threads=args.threads)
    elif preset.core_kind == "apollo_core":
        mnn_model = MNNCoreApollo(separator.model, mnn_path, backend=args.mnn_backend, threads=args.threads)
    elif preset.core_kind == "bandit_core":
        mnn_model = MNNCoreBandit(separator.model, mnn_path, backend=args.mnn_backend, threads=args.threads)
    elif preset.core_kind == "bandit_v2_core":
        mnn_model = MNNCoreBandit(separator.model, mnn_path, backend=args.mnn_backend, threads=args.threads, flatten_channels=True)
    else:
        raise NotImplementedError(f"unsupported core kind: {preset.core_kind}")
    started = time.perf_counter()
    mnn_outputs = demix_with_model(config, mnn_model, mix)
    mnn_seconds = time.perf_counter() - started

    result = {
        "preset": preset.name,
        "audio": str(args.audio),
        "sample_rate": sr,
        "duration_seconds": float(mix.shape[-1] / sr),
        "batch_size": int(config.inference.batch_size),
        "overlap_size": int(config.inference.overlap_size),
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
