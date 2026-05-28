#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import MNN.expr as F
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnn_export.presets import default_out_dir, get_preset, preset_names  # noqa: E402
from tools.mnn_export.roformer_expr_ops import band_split, mask_band, transformer  # noqa: E402
from tools.mnn_export.roformer_mnn import build_separator, prepare_model_for_export, run_mnnconvert, write_metadata  # noqa: E402


class SegmWrapper(torch.nn.Module):
    def __init__(self, final_norm: torch.nn.Module, estimator: torch.nn.Module):
        super().__init__()
        self.final_norm = final_norm
        self.segm = estimator.segm

    def forward(self, x):
        x = self.final_norm(x)
        segm = self.segm(x.permute(0, 3, 1, 2))
        return segm.permute(0, 2, 3, 1).reshape(segm.shape[0], segm.shape[2], -1)


def has_hyperace_segm(model) -> bool:
    return any(hasattr(estimator, "segm") for estimator in getattr(model, "mask_estimators", ()))


def build_manifest(
    preset,
    model,
    exported,
    *,
    band_shape,
    time_shape,
    freq_shape,
    mask_shape,
    batch,
    frames,
    dim,
    time_batch,
    freq_batch,
    segm_shape,
    segm_output_shape,
):
    mask_mode = str(getattr(model, "mask_mode", preset.mask_mode))
    return {
        "preset": preset.name,
        "model_type": type(model).__name__,
        "mask_mode": mask_mode,
        "segments": exported,
        "depth": len(model.layers),
        "num_bands": len(model.band_split.dim_inputs),
        "band_shape": band_shape,
        "time_shape": time_shape,
        "freq_shape": freq_shape,
        "mask_band_input_shape": [batch, frames, dim],
        "segm_input_shape": segm_shape,
        "segm_output_shape": segm_output_shape,
        "has_segm": has_hyperace_segm(model) and mask_mode != "no_segm",
        "has_mask_bands": mask_mode != "segm_only",
        "output_shape": mask_shape,
        "dim_inputs": list(model.band_split.dim_inputs),
        "time_batch": time_batch,
        "freq_batch": freq_batch,
    }


def scan_exported_segments(out_dir: Path):
    return [{"name": path.stem, "path": str(path)} for path in sorted(out_dir.glob("*.mnn"))]


def parse_args():
    parser = argparse.ArgumentParser(description="Export low-memory MNN expr segments for RoFormer inference.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {preset_names()}")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--time-batch", type=int, default=1)
    parser.add_argument("--freq-batch", type=int, default=16)
    parser.add_argument("--only", choices=("all", "transformers", "mask_bands", "segm", "manifest"), default="all")
    return parser.parse_args()


def save_var(var, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    var.name = "output"
    F.save([var], str(path))


def export_transformer(module, shape, path: Path):
    x = F.placeholder(list(shape), F.NCHW, F.float)
    x.name = "input"
    y = transformer(module, x)
    save_var(y, path)
    return list(y.shape)


def export_mask_band(model, band_index: int, shape, path: Path):
    x = F.placeholder(list(shape), F.NCHW, F.float)
    x.name = "input"
    y = mask_band(model, band_index, x)
    save_var(y, path)
    return list(y.shape)


def export_segm(model, stem_index: int, shape, out_dir: Path):
    wrapper = SegmWrapper(model.final_norm, model.mask_estimators[stem_index]).eval()
    x = torch.zeros(*shape, dtype=torch.float32)
    onnx_path = out_dir / f"segm_{stem_index:02d}.onnx"
    mnn_path = out_dir / f"segm_{stem_index:02d}.mnn"
    with torch.inference_mode():
        output = wrapper(x)
        torch.onnx.export(
            wrapper,
            x,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            dynamo=False,
        )
    log = run_mnnconvert(onnx_path, mnn_path, timeout_seconds=1800)
    mnn_path.with_suffix(".mnn.log").write_text(log, encoding="utf-8")
    return list(output.shape)


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    out_dir = args.out_dir / preset.name / "expr_micro_segments"
    separator = build_separator(preset, batch_size=preset.input_shape[0], device="cpu")
    model = prepare_model_for_export(separator.model.cpu())
    mask_mode = str(getattr(model, "mask_mode", preset.mask_mode))
    if preset.input_shape[0] != 1:
        raise ValueError("HyperACE segm MNN export is validated with batch_size=1 only")

    batch, freq_channels, frames, complex_dim = preset.input_shape
    bands = len(model.band_split.dim_inputs)
    dim = int(model.layers[0][0].layers[0][0].norm.gamma.numel())
    band_shape = [batch, frames, bands, dim]
    time_shape = [args.time_batch, frames, dim]
    freq_shape = [args.freq_batch, bands, dim]
    segm_shape = [batch, frames, bands, dim]
    segm_output_shape = [batch, frames, sum(model.band_split.dim_inputs)]
    mask_shape = [batch, len(model.mask_estimators), freq_channels, frames, complex_dim]
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.only == "manifest":
        exported = scan_exported_segments(out_dir)
        manifest = build_manifest(
            preset,
            model,
            exported,
            band_shape=band_shape,
            time_shape=time_shape,
            freq_shape=freq_shape,
            mask_shape=mask_shape,
            batch=batch,
            frames=frames,
            dim=dim,
            time_batch=args.time_batch,
            freq_batch=args.freq_batch,
            segm_shape=segm_shape,
            segm_output_shape=segm_output_shape,
        )
        write_metadata(out_dir.parent / f"{preset.name}_metadata.json", preset, separator, tuple(mask_shape), compact=False)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        separator.del_cache()
        return

    exported = []
    if args.only == "all":
        name = "band_split"
        dst = out_dir / f"{name}.mnn"
        x = F.placeholder(list(preset.input_shape), F.NCHW, F.float)
        x.name = "input"
        y = band_split(model, x)
        save_var(y, dst)
        exported.append({"name": name, "path": str(dst), "input_shape": list(x.shape), "output_shape": list(y.shape)})

    if args.only in ("all", "transformers"):
        for layer_index, (time_transformer, freq_transformer) in enumerate(model.layers):
            name = f"layer_{layer_index:02d}_time"
            exported.append(
                {
                    "name": name,
                    "path": str(out_dir / f"{name}.mnn"),
                    "input_shape": time_shape,
                    "output_shape": export_transformer(time_transformer, time_shape, out_dir / f"{name}.mnn"),
                }
            )
            name = f"layer_{layer_index:02d}_freq"
            exported.append(
                {
                    "name": name,
                    "path": str(out_dir / f"{name}.mnn"),
                    "input_shape": freq_shape,
                    "output_shape": export_transformer(freq_transformer, freq_shape, out_dir / f"{name}.mnn"),
                }
            )

    if args.only in ("all", "mask_bands") and mask_mode != "segm_only":
        for band_index, dim_input in enumerate(model.band_split.dim_inputs):
            name = f"mask_band_{band_index:02d}"
            exported.append(
                {
                    "name": name,
                    "path": str(out_dir / f"{name}.mnn"),
                    "input_shape": [batch, frames, dim],
                    "output_shape": export_mask_band(model, band_index, [batch, frames, dim], out_dir / f"{name}.mnn"),
                    "dim_input": int(dim_input),
                }
            )

    if args.only in ("all", "segm") and has_hyperace_segm(model) and mask_mode != "no_segm":
        for stem_index in range(len(model.mask_estimators)):
            name = f"segm_{stem_index:02d}"
            exported.append(
                {
                    "name": name,
                    "path": str(out_dir / f"{name}.mnn"),
                    "input_shape": segm_shape,
                    "output_shape": export_segm(model, stem_index, segm_shape, out_dir),
                }
            )

    write_metadata(out_dir.parent / f"{preset.name}_metadata.json", preset, separator, tuple(mask_shape), compact=False)
    manifest_segments = exported if args.only == "all" else scan_exported_segments(out_dir)
    manifest = build_manifest(
        preset,
        model,
        manifest_segments,
        band_shape=band_shape,
        time_shape=time_shape,
        freq_shape=freq_shape,
        mask_shape=mask_shape,
        batch=batch,
        frames=frames,
        dim=dim,
        time_batch=args.time_batch,
        freq_batch=args.freq_batch,
        segm_shape=segm_shape,
        segm_output_shape=segm_output_shape,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    separator.del_cache()


if __name__ == "__main__":
    main()
