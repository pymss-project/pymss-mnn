#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnn_export.vr_mnn import (  # noqa: E402
    build_vr_separator,
    default_out_dir,
    get_vr_preset,
    prepare_vr_core_model,
    random_vr_input,
    run_mnnconvert,
    vr_preset_names,
    write_vr_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a VR predict_mask neural core to ONNX/MNN.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {vr_preset_names()}")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-mnn", action="store_true")
    parser.add_argument("--convert-timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_vr_preset(args.preset)
    out_dir = args.out_dir / preset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{preset.name}_core.onnx"
    mnn_path = out_dir / f"{preset.name}_core.mnn"
    metadata_path = out_dir / f"{preset.name}_metadata.json"
    convert_log_path = out_dir / "mnnconvert_core.log"

    separator = build_vr_separator(preset, device="cpu")
    wrapper = prepare_vr_core_model(separator)
    example = random_vr_input(preset)

    with torch.inference_mode():
        output = wrapper(example)
    output_shape = tuple(int(dim) for dim in output.shape)
    write_vr_metadata(metadata_path, preset, separator, output_shape)

    torch.onnx.export(
        wrapper,
        example,
        onnx_path,
        input_names=["mag_patch"],
        output_names=["mask_patch"],
        opset_version=args.opset,
        dynamo=False,
    )

    result = {
        "preset": preset.name,
        "onnx": str(onnx_path),
        "metadata": str(metadata_path),
        "input_shape": list(preset.input_shape),
        "output_shape": list(output_shape),
    }
    if not args.skip_mnn:
        extra_args = ["--useOriginRNNImpl"] if separator.model.is_vr_51_model else None
        log = run_mnnconvert(onnx_path, mnn_path, timeout_seconds=args.convert_timeout, extra_args=extra_args)
        convert_log_path.write_text(log, encoding="utf-8")
        result["mnn"] = str(mnn_path)
        result["mnnconvert_log"] = str(convert_log_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    separator.del_cache()


if __name__ == "__main__":
    main()
