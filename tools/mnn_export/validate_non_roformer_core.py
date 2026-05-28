#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnn_export.non_roformer_mnn import (  # noqa: E402
    build_non_roformer_separator,
    default_out_dir,
    get_non_roformer_preset,
    metrics,
    non_roformer_preset_names,
    prepare_non_roformer_core_model,
    random_inputs,
)
from tools.mnn_export.roformer_mnn import MNNMultiRunner, MNNRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch, ONNX Runtime, and MNN for a non-RoFormer core.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {non_roformer_preset_names()}")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--mnn-backend", default="CPU")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_non_roformer_preset(args.preset)
    out_dir = args.out_dir / preset.name
    onnx_path = out_dir / f"{preset.name}_core.onnx"
    mnn_path = out_dir / f"{preset.name}_core.mnn"
    if not onnx_path.exists():
        raise FileNotFoundError(f"missing ONNX model: {onnx_path}")
    if not mnn_path.exists():
        raise FileNotFoundError(f"missing MNN model: {mnn_path}")

    separator = build_non_roformer_separator(preset, batch_size=1, device="cpu")
    wrapper = prepare_non_roformer_core_model(separator, preset)
    inputs = random_inputs(preset, seed=args.seed)
    input_np = {
        name: tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, tensor in zip(preset.input_names, inputs)
    }

    with torch.inference_mode():
        torch_outputs = wrapper(*inputs)
    if not isinstance(torch_outputs, tuple):
        torch_outputs = (torch_outputs,)
    torch_np = [output.detach().cpu().numpy().astype(np.float32, copy=False) for output in torch_outputs]

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = ort_session.run(list(preset.output_names), input_np)

    if len(preset.input_names) == 1 and len(preset.output_names) == 1:
        runner = MNNRunner(mnn_path, input_name=preset.input_names[0], output_name=preset.output_names[0], backend=args.mnn_backend, threads=args.threads)
        mnn_outputs = {preset.output_names[0]: runner(input_np[preset.input_names[0]])}
    else:
        runner = MNNMultiRunner(
            mnn_path,
            input_names=preset.input_names,
            output_names=preset.output_names,
            backend=args.mnn_backend,
            threads=args.threads,
        )
        mnn_outputs = runner(dict(input_np))

    result = {
        "preset": preset.name,
        "input_shapes": {name: list(value.shape) for name, value in input_np.items()},
        "outputs": {},
    }
    for index, name in enumerate(preset.output_names):
        candidate = mnn_outputs.get(name)
        result["outputs"][name] = {
            "torch_shape": list(torch_np[index].shape),
            "onnx_vs_torch": metrics(torch_np[index], ort_outputs[index]),
            "mnn_vs_torch": metrics(torch_np[index], candidate) if candidate is not None else None,
            "mnn_vs_onnx": metrics(ort_outputs[index], candidate) if candidate is not None else None,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    separator.del_cache()


if __name__ == "__main__":
    main()
