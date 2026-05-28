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

from tools.mnn_export.roformer_mnn import MNNRunner  # noqa: E402
from tools.mnn_export.vr_mnn import (  # noqa: E402
    build_vr_separator,
    default_out_dir,
    get_vr_preset,
    metrics,
    prepare_vr_core_model,
    random_vr_input,
    vr_preset_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch, ONNX Runtime, and MNN for a VR core.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {vr_preset_names()}")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--mnn-backend", default="CPU")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_vr_preset(args.preset)
    out_dir = args.out_dir / preset.name
    onnx_path = out_dir / f"{preset.name}_core.onnx"
    mnn_path = out_dir / f"{preset.name}_core.mnn"
    if not onnx_path.exists():
        raise FileNotFoundError(f"missing ONNX model: {onnx_path}")
    if not mnn_path.exists():
        raise FileNotFoundError(f"missing MNN model: {mnn_path}")

    separator = build_vr_separator(preset, device="cpu")
    wrapper = prepare_vr_core_model(separator)
    x = random_vr_input(preset, seed=args.seed)
    x_np = x.detach().cpu().numpy().astype(np.float32, copy=False)

    with torch.inference_mode():
        torch_out = wrapper(x).detach().cpu().numpy().astype(np.float32, copy=False)

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = ort_session.run(["mask_patch"], {"mag_patch": x_np})[0]

    runner = MNNRunner(mnn_path, input_name="mag_patch", output_name="mask_patch", backend=args.mnn_backend, threads=args.threads)
    mnn_out = runner(x_np)

    print(json.dumps(
        {
            "preset": preset.name,
            "input_shape": list(x_np.shape),
            "output_shape": list(torch_out.shape),
            "onnx_vs_torch": metrics(torch_out, ort_out),
            "mnn_vs_torch": metrics(torch_out, mnn_out),
            "mnn_vs_onnx": metrics(ort_out, mnn_out),
        },
        ensure_ascii=False,
        indent=2,
    ))
    separator.del_cache()


if __name__ == "__main__":
    main()
