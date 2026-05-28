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

from tools.mnn_export.non_roformer_mnn import (  # noqa: E402
    build_non_roformer_separator,
    default_out_dir,
    get_non_roformer_preset,
    non_roformer_preset_names,
    prepare_non_roformer_core_model,
    random_inputs,
    run_mnnconvert,
    write_non_roformer_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a non-RoFormer neural core to ONNX/MNN.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {non_roformer_preset_names()}")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-mnn", action="store_true")
    parser.add_argument("--convert-timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_non_roformer_preset(args.preset)
    out_dir = args.out_dir / preset.name
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{preset.name}_core.onnx"
    mnn_path = out_dir / f"{preset.name}_core.mnn"
    metadata_path = out_dir / f"{preset.name}_metadata.json"
    convert_log_path = out_dir / "mnnconvert_core.log"

    separator = build_non_roformer_separator(preset, batch_size=1, device="cpu")
    wrapper = prepare_non_roformer_core_model(separator, preset)
    examples = random_inputs(preset)

    with torch.inference_mode():
        outputs = wrapper(*examples)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    output_shapes = tuple(tuple(int(dim) for dim in output.shape) for output in outputs)
    write_non_roformer_metadata(metadata_path, preset, separator, output_shapes)

    torch.onnx.export(
        wrapper,
        examples if len(examples) > 1 else examples[0],
        onnx_path,
        input_names=list(preset.input_names),
        output_names=list(preset.output_names),
        opset_version=args.opset,
        dynamo=False,
    )

    result = {
        "preset": preset.name,
        "onnx": str(onnx_path),
        "metadata": str(metadata_path),
        "input_shapes": [list(shape) for shape in preset.input_shapes],
        "output_shapes": [list(shape) for shape in output_shapes],
    }
    if not args.skip_mnn:
        extra_args = ["--useOriginRNNImpl"] if preset.core_kind in {"scnet_core", "bandit_core", "bandit_v2_core"} else None
        log = run_mnnconvert(onnx_path, mnn_path, timeout_seconds=args.convert_timeout, extra_args=extra_args)
        convert_log_path.write_text(log, encoding="utf-8")
        result["mnn"] = str(mnn_path)
        result["mnnconvert_log"] = str(convert_log_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    separator.del_cache()


if __name__ == "__main__":
    main()
