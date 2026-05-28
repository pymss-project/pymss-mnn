#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymss.audio_io import save_audio  # noqa: E402
from tools.mnn_export.roformer_mnn import MNNRunner  # noqa: E402
from tools.mnn_export.vr_mnn import (  # noqa: E402
    audio_metrics_by_stem,
    build_vr_separator,
    default_out_dir,
    get_vr_preset,
    load_vr_audio,
    separate_vr_with_runner,
    vr_preset_names,
    vr_torch_runner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end VR separation with an MNN predict_mask core.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {vr_preset_names()}")
    parser.add_argument("--audio", type=Path, default=ROOT / "test.m4a")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--store-dir", type=Path, default=None)
    parser.add_argument("--compare-torch", action="store_true")
    parser.add_argument("--mnn-backend", default="CPU")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=None)
    return parser.parse_args()


def save_outputs(store_dir: Path, outputs: dict[str, np.ndarray], sample_rate: int, stem: str) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)
    params = {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k", "m4a_bit_rate": "192k", "m4a_aac_at_quality": 2}
    for name, audio in outputs.items():
        safe_name = name.replace("/", "_").replace(" ", "_")
        save_audio(str(store_dir / f"{stem}_{safe_name}.wav"), np.asarray(audio), sample_rate, "wav", params)


def main() -> None:
    args = parse_args()
    preset = get_vr_preset(args.preset)
    mnn_path = args.out_dir / preset.name / f"{preset.name}_core.mnn"
    if not mnn_path.exists():
        raise FileNotFoundError(f"missing MNN model: {mnn_path}")

    separator = build_vr_separator(preset, device="cpu")
    model = separator.model
    duration = args.seconds if args.seconds and args.seconds > 0 else None
    mix, sr = load_vr_audio(args.audio, sample_rate=44100, seconds=duration)

    runner = MNNRunner(mnn_path, input_name="mag_patch", output_name="mask_patch", backend=args.mnn_backend, threads=args.threads)
    started = time.perf_counter()
    mnn_outputs = separate_vr_with_runner(model, mix, sr, runner)
    mnn_seconds = time.perf_counter() - started

    result = {
        "preset": preset.name,
        "audio": str(args.audio),
        "sample_rate": sr,
        "duration_seconds": float(mix.shape[-1] / sr),
        "batch_size": int(model.batch_size),
        "window_size": int(model.window_size),
        "mnn_seconds": mnn_seconds,
        "outputs": {name: list(value.shape) for name, value in mnn_outputs.items()},
    }
    if args.compare_torch:
        started = time.perf_counter()
        torch_outputs = separate_vr_with_runner(model, mix, sr, vr_torch_runner(model))
        torch_seconds = time.perf_counter() - started
        result["torch_seconds"] = torch_seconds
        result["mnn_vs_torch"] = audio_metrics_by_stem(torch_outputs, mnn_outputs)

    store_dir = args.store_dir or (args.out_dir / preset.name / "audio_outputs")
    save_outputs(store_dir, mnn_outputs, 44100, args.audio.stem)
    result["store_dir"] = str(store_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    separator.del_cache()


if __name__ == "__main__":
    main()
