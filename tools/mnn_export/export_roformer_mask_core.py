from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnn_export.presets import apply_shape_overrides, default_out_dir, get_preset, preset_names  # noqa: E402
from tools.mnn_export.roformer_mnn import (  # noqa: E402
    build_separator,
    patch_rotary_for_export,
    prepare_mask_core_model,
    random_stft_input,
    restore_rotary,
    run_mnnconvert,
    write_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fixed-shape RoFormer mask core as one MNN graph.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {preset_names()}")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--seed", type=int, default=1145)
    parser.add_argument("--convert-timeout", type=int, default=2400)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override the fixed MNN STFT frame count for a mobile/lower-memory export.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override metadata/inference chunk size. If --frames is omitted, frames are derived from this value.",
    )
    parser.add_argument("--overlap-size", type=int, default=None, help="Override metadata/inference overlap size.")
    parser.add_argument("--variant-name", default=None, help="Output preset name for custom shape exports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = apply_shape_overrides(
        get_preset(args.preset),
        frames=args.frames,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        variant_name=args.variant_name,
    )
    out_dir = args.out_dir / preset.name
    out_dir.mkdir(parents=True, exist_ok=True)

    separator = build_separator(preset, batch_size=1, device="cpu")
    wrapper = prepare_mask_core_model(separator)
    example = random_stft_input(tuple(preset.input_shape), seed=args.seed)

    onnx_path = out_dir / f"{preset.name}_mask_core.onnx"
    mnn_path = out_dir / f"{preset.name}_mask_core.mnn"
    metadata_path = out_dir / f"{preset.name}_metadata.json"

    previous_rotary = patch_rotary_for_export()
    try:
        with torch.inference_mode():
            output = wrapper(example)
            torch.onnx.export(
                wrapper,
                example,
                onnx_path,
                input_names=["stft_repr"],
                output_names=["mask"],
                opset_version=17,
                dynamo=False,
            )
    finally:
        restore_rotary(previous_rotary)

    write_metadata(metadata_path, preset, separator, tuple(output.shape), compact=False)
    log = run_mnnconvert(onnx_path, mnn_path, timeout_seconds=args.convert_timeout)
    mnn_path.with_suffix(".mnn.log").write_text(log, encoding="utf-8")
    print(mnn_path)


if __name__ == "__main__":
    main()
