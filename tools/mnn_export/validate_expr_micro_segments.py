#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnn_export.presets import default_out_dir, get_preset, preset_names  # noqa: E402
from tools.mnn_export.roformer_mnn import MNNRunner, build_separator, metrics, prepare_mask_core_model, random_stft_input  # noqa: E402


class ExprMicroRuntime:
    def __init__(self, segment_dir: Path, depth: int, *, time_batch: int, freq_batch: int, backend: str = "CPU", threads: int = 1, mask_head=None):
        self.segment_dir = Path(segment_dir)
        self.depth = depth
        self.time_batch = time_batch
        self.freq_batch = freq_batch
        self.backend = backend
        self.threads = threads
        self.mask_head = mask_head
        self._runners = {}
        manifest_path = self.segment_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        self.dim_inputs = tuple(int(value) for value in self.manifest.get("dim_inputs", ()))
        self.mask_mode = str(self.manifest.get("mask_mode", "no_segm"))
        self.transformer_split = str(self.manifest.get("transformer_split", "fused"))
        self.transformer_block_size = max(0, int(self.manifest.get("transformer_block_size", 0)))
        self.transformer_block_count = max(0, int(self.manifest.get("transformer_block_count", 0)))
        if self.transformer_block_size and self.transformer_block_count == 0:
            self.transformer_block_count = (depth + self.transformer_block_size - 1) // self.transformer_block_size
        self.mask_group_size = max(1, int(self.manifest.get("mask_group_size", 1)))
        manifest_depth = self.manifest.get("depth")
        if manifest_depth is not None and int(manifest_depth) != depth:
            raise ValueError(f"manifest depth {manifest_depth} does not match runtime depth {depth}")

    @property
    def mask_head_kind(self) -> str:
        if self.mask_head is not None:
            return "torch_mask_head"
        if self.mask_mode == "segm_only" and (self.segment_dir / "segm_00.mnn").exists():
            return "mnn_segm"
        if self.mask_mode == "full" and (self.segment_dir / "segm_00.mnn").exists():
            return "mnn_mask_bands_plus_segm"
        if self.dim_inputs and (self.segment_dir / "mask_group_00.mnn").exists():
            return "mnn_mask_groups"
        if self.dim_inputs and (self.segment_dir / "mask_band_00.mnn").exists():
            return "mnn_mask_bands"
        return "mnn_mask_head"

    def run(self, name: str, x: np.ndarray) -> np.ndarray:
        runner = self._runners.get(name)
        if runner is None:
            runner = MNNRunner(self.segment_dir / f"{name}.mnn", input_name="input", output_name="output", backend=self.backend, threads=self.threads)
            self._runners[name] = runner
        return runner(x)

    def run_transformer(self, name: str, x: np.ndarray) -> np.ndarray:
        if self.transformer_split == "attention_ffn":
            return self.run(f"{name}_ffn", self.run(f"{name}_attn", x))
        return self.run(name, x)

    def run_time(self, layer_index: int, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        batch, frames, bands, dim = x.shape
        flat = x.transpose(0, 2, 1, 3).reshape(batch * bands, frames, dim)
        flat_out = np.empty_like(flat)
        for start in range(0, flat.shape[0], self.time_batch):
            end = min(start + self.time_batch, flat.shape[0])
            chunk = flat[start:end]
            if chunk.shape[0] != self.time_batch:
                pad = np.zeros((self.time_batch - chunk.shape[0], frames, dim), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
                flat_out[start:end] = self.run_transformer(f"layer_{layer_index:02d}_time", chunk)[: end - start]
            else:
                flat_out[start:end] = self.run_transformer(f"layer_{layer_index:02d}_time", chunk)
        out[...] = flat_out.reshape(batch, bands, frames, dim).transpose(0, 2, 1, 3)
        return out

    def run_freq(self, layer_index: int, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        batch, frames, bands, dim = x.shape
        flat = x.reshape(batch * frames, bands, dim)
        flat_out = np.empty_like(flat)
        for start in range(0, flat.shape[0], self.freq_batch):
            end = min(start + self.freq_batch, flat.shape[0])
            chunk = flat[start:end]
            if chunk.shape[0] != self.freq_batch:
                pad = np.zeros((self.freq_batch - chunk.shape[0], bands, dim), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
                flat_out[start:end] = self.run_transformer(f"layer_{layer_index:02d}_freq", chunk)[: end - start]
            else:
                flat_out[start:end] = self.run_transformer(f"layer_{layer_index:02d}_freq", chunk)
        out[...] = flat_out.reshape(batch, frames, bands, dim)
        return out

    def run_transformer_blocks(self, x: np.ndarray) -> np.ndarray:
        for block_index in range(self.transformer_block_count):
            name = f"block_{block_index:02d}"
            x = np.ascontiguousarray(self.run(name, x))
            self._runners.pop(name, None)
            gc.collect()
        return x

    def __call__(self, stft_repr: np.ndarray) -> np.ndarray:
        x = self.run("band_split", stft_repr)
        if self.transformer_block_size:
            x = self.run_transformer_blocks(x)
        else:
            for layer_index in range(self.depth):
                x = np.ascontiguousarray(self.run_time(layer_index, x))
                x = np.ascontiguousarray(self.run_freq(layer_index, x))
        if self.mask_head is not None:
            with torch.inference_mode():
                return self.mask_head(torch.from_numpy(x)).detach().cpu().numpy()
        if self.dim_inputs and (
            (self.segment_dir / "mask_group_00.mnn").exists()
            or (self.segment_dir / "mask_band_00.mnn").exists()
            or (self.mask_mode == "segm_only" and (self.segment_dir / "segm_00.mnn").exists())
        ):
            return self.run_mask_bands(x)
        return self.run("mask_head", x)

    def _run_grouped_mask_bands(self, x: np.ndarray) -> np.ndarray:
        batch, frames, bands, dim = x.shape
        flat_dim = sum(self.dim_inputs)
        output = np.zeros((batch, self.manifest["output_shape"][1], frames, flat_dim), dtype=np.float32)
        offset = 0
        group_index = 0
        for band_start in range(0, bands, self.mask_group_size):
            band_count = min(self.mask_group_size, bands - band_start)
            group_dims = self.dim_inputs[band_start : band_start + band_count]
            group_width = sum(group_dims)
            group_x = np.ascontiguousarray(x[:, :, band_start : band_start + band_count, :])
            group_out = self.run(f"mask_group_{group_index:02d}", group_x)
            expected = (batch, output.shape[1], frames, group_width)
            if tuple(group_out.shape) != expected:
                raise ValueError(f"mask group output shape {tuple(group_out.shape)} does not match {expected}")
            output[:, :, :, offset : offset + group_width] = group_out
            offset += group_width
            group_index += 1
        if offset != flat_dim:
            raise ValueError(f"grouped mask width {offset} does not match sum(dim_inputs) {flat_dim}")
        return output

    def run_mask_bands(self, x: np.ndarray) -> np.ndarray:
        if not self.dim_inputs:
            raise ValueError("manifest is missing dim_inputs for per-band mask execution")
        if len(self.dim_inputs) != x.shape[2]:
            raise ValueError(f"dim_inputs length {len(self.dim_inputs)} does not match band count {x.shape[2]}")
        if self.mask_mode != "segm_only" and self.mask_group_size > 1:
            flat = self._run_grouped_mask_bands(x)
        else:
            flat = None
            if self.mask_mode != "segm_only":
                for band_index in range(x.shape[2]):
                    path = self.segment_dir / f"mask_band_{band_index:02d}.mnn"
                    if not path.exists():
                        raise FileNotFoundError(f"missing MNN mask band segment: {path}")
                band_outputs = []
                for band_index in range(x.shape[2]):
                    band_x = np.ascontiguousarray(x[:, :, band_index, :])
                    band_outputs.append(self.run(f"mask_band_{band_index:02d}", band_x))
                flat = np.concatenate(band_outputs, axis=-1)

        if self.mask_mode != "no_segm":
            expected_shape = self.manifest.get("output_shape")
            stem_count = int(expected_shape[1]) if expected_shape is not None else (flat.shape[1] if flat is not None else 0)
            if stem_count <= 0:
                raise ValueError("cannot infer stem count for segm MNN execution")
            segm_outputs = []
            segm_input = np.ascontiguousarray(x)
            for stem_index in range(stem_count):
                path = self.segment_dir / f"segm_{stem_index:02d}.mnn"
                if not path.exists():
                    raise FileNotFoundError(f"missing MNN segm segment: {path}")
                segm_outputs.append(self.run(f"segm_{stem_index:02d}", segm_input))
            segm_flat = np.stack(segm_outputs, axis=1)
            flat = segm_flat if flat is None else flat + segm_flat

        if flat is None:
            raise ValueError("no MNN mask output segments were executed")
        if flat.ndim == 5:
            output = flat
        else:
            batch, stems, frames, flat_dim = flat.shape
            if flat_dim != sum(self.dim_inputs):
                raise ValueError(f"concatenated mask width {flat_dim} does not match sum(dim_inputs) {sum(self.dim_inputs)}")
            if flat_dim % 2:
                raise ValueError(f"concatenated mask width must be even, got {flat_dim}")
            output = flat.reshape(batch, stems, frames, flat_dim // 2, 2).transpose(0, 1, 3, 2, 4).copy()
        expected_shape = self.manifest.get("output_shape")
        if expected_shape is not None and list(output.shape) != list(expected_shape):
            raise ValueError(f"mask output shape {list(output.shape)} does not match manifest output_shape {expected_shape}")
        return output

def parse_args():
    parser = argparse.ArgumentParser(description="Validate low-memory expr micro-segmented MNN mask core.")
    parser.add_argument("--preset", required=True, help=f"Preset name: {preset_names()}")
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mnn-backend", default="CPU")
    parser.add_argument("--time-batch", type=int, default=1)
    parser.add_argument("--freq-batch", type=int, default=16)
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument(
        "--torch-mask-head",
        action="store_true",
        help="Run the large final mask head in PyTorch while MNN runs band split and transformer blocks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = get_preset(args.preset)
    depth = 12
    x = random_stft_input(preset.input_shape, args.seed)
    ref_path = None
    separator = None
    wrapper = None
    if not args.skip_torch or args.torch_mask_head:
        separator = build_separator(preset, batch_size=preset.input_shape[0], device="cpu")
        wrapper = prepare_mask_core_model(separator)
        depth = len(separator.model.layers)

    if not args.skip_torch:
        with torch.inference_mode():
            torch_out = wrapper(x).detach().cpu().numpy()
        ref_path = Path(tempfile.gettempdir()) / f"pymss_{preset.name}_torch_micro_ref.npy"
        np.save(ref_path, torch_out)
        del torch_out
        gc.collect()
    mask_head = None
    if args.torch_mask_head:
        from tools.mnn_export.roformer_mnn import MaskHeadWrapper

        mask_head = MaskHeadWrapper(separator.model).eval()
    if separator is not None and not args.torch_mask_head:
        separator.del_cache()
        del wrapper
        separator = None
        gc.collect()

    runtime = ExprMicroRuntime(
        args.out_dir / preset.name / "expr_micro_segments",
        depth=depth,
        time_batch=args.time_batch,
        freq_batch=args.freq_batch,
        backend=args.mnn_backend,
        threads=args.threads,
        mask_head=mask_head,
    )
    mnn_out = runtime(x.detach().cpu().numpy())

    result = {
        "preset": preset.name,
        "input_shape": list(x.shape),
        "output_shape": list(mnn_out.shape),
        "mask_head": runtime.mask_head_kind,
    }
    if ref_path is not None:
        torch_out = np.load(ref_path, mmap_mode="r")
        result["mnn_vs_torch"] = metrics(torch_out, mnn_out)
    if separator is not None:
        separator.del_cache()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
