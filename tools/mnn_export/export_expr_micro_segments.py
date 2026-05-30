#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import MNN.expr as F
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnn_export.presets import apply_shape_overrides, default_out_dir, get_preset, preset_names  # noqa: E402
from tools.mnn_export.roformer_expr_ops import (  # noqa: E402
    band_split,
    const_cache,
    mask_core,
    mask_band,
    mask_band_group,
    roformer_block,
    roformer_block_grouped,
    roformer_block_unrolled,
    transformer,
    transformer_attention_block,
    transformer_ffn_block,
)
from tools.mnn_export.roformer_mnn import (  # noqa: E402
    build_separator,
    mnnconvert_binary,
    prepare_model_for_export,
    run_mnnconvert,
    write_metadata,
)


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
    attention_op,
    transformer_split,
    transformer_block_size,
    transformer_block_mode,
    transformer_block_time_group_size,
):
    mask_mode = str(getattr(model, "mask_mode", preset.mask_mode))
    transformer_block_size = max(0, int(transformer_block_size))
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
        "mask_group_size": max(1, int(getattr(model, "_pymss_mnn_mask_group_size", 1))),
        "attention_op": attention_op,
        "transformer_split": transformer_split,
        "transformer_block_size": transformer_block_size,
        "transformer_block_count": (len(model.layers) + transformer_block_size - 1) // transformer_block_size if transformer_block_size else 0,
        "transformer_block_mode": transformer_block_mode if transformer_block_size else "",
        "transformer_block_time_group_size": int(transformer_block_time_group_size) if transformer_block_size else 0,
    }


def scan_exported_segments(out_dir: Path):
    return [{"name": path.stem, "path": str(path)} for path in sorted(out_dir.glob("*.mnn"))]


def parse_args():
    parser = argparse.ArgumentParser(description="Export low-memory MNN expr segments for RoFormer inference.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {preset_names()}")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument(
        "--time-batch",
        type=int,
        default=1,
        help="Batch independent time-transformer band sequences. Keep 1 for validated Metal exports.",
    )
    parser.add_argument("--freq-batch", type=int, default=16)
    parser.add_argument("--only", choices=("all", "transformers", "mask_bands", "segm", "manifest", "core_model"), default="all")
    parser.add_argument(
        "--mask-group-size",
        type=int,
        default=1,
        help="Export mask estimators in groups of N bands to reduce runtime sessions. 1 keeps the lowest-memory per-band layout.",
    )
    parser.add_argument(
        "--attention-op",
        choices=("manual", "mnn", "fmha_v2", "fmha_v2_gate"),
        default="fmha_v2",
        help="Use legacy MatMul/Softmax, native MNN Attention, packed FmhaV2, or experimental packed qkv+gate FmhaV2 for the forked Metal backend.",
    )
    parser.add_argument(
        "--transformer-split",
        choices=("fused", "attention_ffn"),
        default="fused",
        help="Export each transformer as one segment or split it into attention and FFN segments for segment-level precision control.",
    )
    parser.add_argument(
        "--transformer-block-size",
        type=int,
        default=6,
        help="Export whole RoFormer layer blocks as MNN Expr segments. 6 is the mobile-first default; 0 keeps legacy per time/freq transformer segments.",
    )
    parser.add_argument(
        "--transformer-block-mode",
        choices=("grouped", "batched", "unrolled"),
        default="batched",
        help="batched is the validated block default; grouped/unrolled are diagnostic fallbacks and can create much larger graphs.",
    )
    parser.add_argument(
        "--transformer-block-time-group-size",
        type=int,
        default=8,
        help="Band group size for --transformer-block-mode grouped. Lower values reduce peak memory and increase graph size.",
    )
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


def translate_json_ops(path: Path) -> None:
    raw_path = path.with_suffix(".jsonop.mnn")
    path.replace(raw_path)
    cmd = [
        mnnconvert_binary(),
        "-f",
        "MNN",
        "--optimizeLevel",
        "1",
        "--modelFile",
        str(raw_path),
        "--MNNModel",
        str(path),
        "--bizCode",
        "pymss",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800)
    path.with_suffix(".mnn.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"MNNConvert JSON op translation failed with code {proc.returncode}\n{proc.stdout}")
    raw_path.unlink(missing_ok=True)


def save_var(var, path: Path, *, translate_json: bool = False, output_name: str = "output") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    var.name = output_name
    F.save([var], str(path))
    if translate_json:
        translate_json_ops(path)


def export_transformer(module, shape, path: Path, *, attention_op: str):
    x = F.placeholder(list(shape), F.NCHW, F.float)
    x.name = "input"
    y = transformer(module, x, attention_op=attention_op)
    save_var(y, path, translate_json=attention_op in ("mnn", "fmha_v2", "fmha_v2_gate"))
    return list(y.shape) if y.shape is not None else list(shape)


def export_transformer_split(module, shape, path_prefix: Path, *, attention_op: str):
    x = F.placeholder(list(shape), F.NCHW, F.float)
    x.name = "input"
    y = transformer_attention_block(module, x, attention_op=attention_op)
    save_var(y, path_prefix.with_name(path_prefix.name + "_attn.mnn"), translate_json=attention_op in ("mnn", "fmha_v2", "fmha_v2_gate"))

    x = F.placeholder(list(shape), F.NCHW, F.float)
    x.name = "input"
    y = transformer_ffn_block(module, x)
    save_var(y, path_prefix.with_name(path_prefix.name + "_ffn.mnn"))
    return list(y.shape) if y.shape is not None else list(shape)


def export_mask_band(model, band_index: int, shape, path: Path):
    x = F.placeholder(list(shape), F.NCHW, F.float)
    x.name = "input"
    y = mask_band(model, band_index, x)
    save_var(y, path)
    return list(y.shape)


def export_mask_group(model, band_start: int, band_count: int, shape, path: Path):
    x = F.placeholder(list(shape), F.NCHW, F.float)
    x.name = "input"
    y = mask_band_group(model, band_start, band_count, x)
    save_var(y, path)
    return list(y.shape)


def export_transformer_block(
    model,
    start_layer: int,
    end_layer: int,
    shape,
    path: Path,
    *,
    freq_batch: int,
    time_group_size: int,
    attention_op: str,
    mode: str,
):
    x = F.placeholder(list(shape), F.NCHW, F.float)
    x.name = "input"
    with const_cache():
        if mode == "grouped":
            y = roformer_block_grouped(model, start_layer, end_layer, x, time_group_size=time_group_size, attention_op=attention_op)
        elif mode == "batched":
            y = roformer_block(model, start_layer, end_layer, x, attention_op=attention_op)
        elif mode == "unrolled":
            y = roformer_block_unrolled(model, start_layer, end_layer, x, freq_batch=freq_batch, attention_op=attention_op)
        else:
            raise ValueError(f"unsupported transformer block mode: {mode}")
    save_var(y, path, translate_json=attention_op in ("mnn", "fmha_v2", "fmha_v2_gate"))
    return list(y.shape) if y.shape is not None else list(shape)


def export_core_model(
    model,
    input_shape,
    path: Path,
    *,
    attention_op: str,
    transformer_block_size: int,
    transformer_block_mode: str,
    transformer_block_time_group_size: int,
    freq_batch: int,
    mask_group_size: int,
):
    x = F.placeholder(list(input_shape), F.NCHW, F.float)
    x.name = "stft_repr"
    with const_cache():
        y = mask_core(
            model,
            x,
            attention_op=attention_op,
            transformer_block_size=transformer_block_size,
            transformer_block_mode=transformer_block_mode,
            transformer_block_time_group_size=transformer_block_time_group_size,
            freq_batch=freq_batch,
            mask_group_size=mask_group_size,
        )
    save_var(y, path, translate_json=attention_op in ("mnn", "fmha_v2", "fmha_v2_gate"), output_name="mask")
    batch, freq_channels, frames, complex_dim = list(input_shape)
    return [batch, len(model.mask_estimators), freq_channels, frames, complex_dim]


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
    preset = apply_shape_overrides(
        get_preset(args.preset),
        frames=args.frames,
        chunk_size=args.chunk_size,
        overlap_size=args.overlap_size,
        variant_name=args.variant_name,
    )
    out_dir = args.out_dir / preset.name / "expr_micro_segments"
    separator = build_separator(preset, batch_size=preset.input_shape[0], device="cpu")
    model = prepare_model_for_export(separator.model.cpu())
    mask_mode = str(getattr(model, "mask_mode", preset.mask_mode))
    if preset.input_shape[0] != 1:
        raise ValueError("HyperACE segm MNN export is validated with batch_size=1 only")
    if args.transformer_split == "attention_ffn" and args.attention_op not in ("mnn", "fmha_v2", "fmha_v2_gate"):
        raise ValueError("--transformer-split attention_ffn is intended for native MNN attention ops")
    if args.transformer_block_size < 0:
        raise ValueError("--transformer-block-size must be >= 0")
    if args.transformer_block_size and args.transformer_split != "fused":
        raise ValueError("--transformer-block-size exports whole blocks and cannot be combined with --transformer-split attention_ffn")
    if args.transformer_block_time_group_size < 1:
        raise ValueError("--transformer-block-time-group-size must be >= 1")
    if args.mask_group_size < 1:
        raise ValueError("--mask-group-size must be >= 1")
    model._pymss_mnn_mask_group_size = int(args.mask_group_size)

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

    if args.only == "core_model":
        core_path = out_dir.parent / f"{preset.name}_expr_mask_core.mnn"
        output_shape = export_core_model(
            model,
            preset.input_shape,
            core_path,
            attention_op=args.attention_op,
            transformer_block_size=args.transformer_block_size,
            transformer_block_mode=args.transformer_block_mode,
            transformer_block_time_group_size=args.transformer_block_time_group_size,
            freq_batch=args.freq_batch,
            mask_group_size=args.mask_group_size,
        )
        write_metadata(out_dir.parent / f"{preset.name}_metadata.json", preset, separator, tuple(output_shape), compact=False)
        print(core_path)
        separator.del_cache()
        return

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
            attention_op=args.attention_op,
            transformer_split=args.transformer_split,
            transformer_block_size=args.transformer_block_size,
            transformer_block_mode=args.transformer_block_mode,
            transformer_block_time_group_size=args.transformer_block_time_group_size,
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
        if args.transformer_block_size:
            block_index = 0
            for start_layer in range(0, len(model.layers), args.transformer_block_size):
                end_layer = min(start_layer + args.transformer_block_size, len(model.layers))
                name = f"block_{block_index:02d}"
                output_shape = export_transformer_block(
                    model,
                    start_layer,
                    end_layer,
                    band_shape,
                    out_dir / f"{name}.mnn",
                    freq_batch=args.freq_batch,
                    time_group_size=args.transformer_block_time_group_size,
                    attention_op=args.attention_op,
                    mode=args.transformer_block_mode,
                )
                exported.append(
                    {
                        "name": name,
                        "path": str(out_dir / f"{name}.mnn"),
                        "input_shape": band_shape,
                        "output_shape": output_shape,
                        "start_layer": start_layer,
                        "end_layer": end_layer,
                    }
                )
                block_index += 1
        else:
            for layer_index, (time_transformer, freq_transformer) in enumerate(model.layers):
                name = f"layer_{layer_index:02d}_time"
                if args.transformer_split == "attention_ffn":
                    output_shape = export_transformer_split(
                        time_transformer,
                        time_shape,
                        out_dir / name,
                        attention_op=args.attention_op,
                    )
                    exported.extend(
                        [
                            {"name": f"{name}_attn", "path": str(out_dir / f"{name}_attn.mnn"), "input_shape": time_shape, "output_shape": output_shape},
                            {"name": f"{name}_ffn", "path": str(out_dir / f"{name}_ffn.mnn"), "input_shape": time_shape, "output_shape": output_shape},
                        ]
                    )
                else:
                    exported.append(
                        {
                            "name": name,
                            "path": str(out_dir / f"{name}.mnn"),
                            "input_shape": time_shape,
                            "output_shape": export_transformer(
                                time_transformer,
                                time_shape,
                                out_dir / f"{name}.mnn",
                                attention_op=args.attention_op,
                            ),
                        }
                    )
                name = f"layer_{layer_index:02d}_freq"
                if args.transformer_split == "attention_ffn":
                    output_shape = export_transformer_split(
                        freq_transformer,
                        freq_shape,
                        out_dir / name,
                        attention_op=args.attention_op,
                    )
                    exported.extend(
                        [
                            {"name": f"{name}_attn", "path": str(out_dir / f"{name}_attn.mnn"), "input_shape": freq_shape, "output_shape": output_shape},
                            {"name": f"{name}_ffn", "path": str(out_dir / f"{name}_ffn.mnn"), "input_shape": freq_shape, "output_shape": output_shape},
                        ]
                    )
                else:
                    exported.append(
                        {
                            "name": name,
                            "path": str(out_dir / f"{name}.mnn"),
                            "input_shape": freq_shape,
                            "output_shape": export_transformer(
                                freq_transformer,
                                freq_shape,
                                out_dir / f"{name}.mnn",
                                attention_op=args.attention_op,
                            ),
                        }
                    )

    if args.only in ("all", "mask_bands") and mask_mode != "segm_only":
        if args.mask_group_size == 1:
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
        else:
            for group_index, band_start in enumerate(range(0, bands, args.mask_group_size)):
                band_count = min(args.mask_group_size, bands - band_start)
                name = f"mask_group_{group_index:02d}"
                dim_inputs = [int(value) for value in model.band_split.dim_inputs[band_start : band_start + band_count]]
                exported.append(
                    {
                        "name": name,
                        "path": str(out_dir / f"{name}.mnn"),
                        "input_shape": [batch, frames, band_count, dim],
                        "output_shape": export_mask_group(
                            model,
                            band_start,
                            band_count,
                            [batch, frames, band_count, dim],
                            out_dir / f"{name}.mnn",
                        ),
                        "band_start": band_start,
                        "band_count": band_count,
                        "dim_inputs": dim_inputs,
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
        attention_op=args.attention_op,
        transformer_split=args.transformer_split,
        transformer_block_size=args.transformer_block_size,
        transformer_block_mode=args.transformer_block_mode,
        transformer_block_time_group_size=args.transformer_block_time_group_size,
    )
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    separator.del_cache()


if __name__ == "__main__":
    main()
