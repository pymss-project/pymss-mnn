from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymss.logger import get_separation_logger, set_log_level  # noqa: E402
from pymss.modules.bs_roformer import transformer as transformer_mod  # noqa: E402
from pymss.modules.bs_roformer.common import stft_roformer  # noqa: E402
from pymss.separator import MSSeparator  # noqa: E402

from tools.mnn_export.presets import RoformerMNNPreset  # noqa: E402


class MaskCoreWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, stft_repr):
        return self.model._forward_mask_core(stft_repr)


class BandSplitWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, stft_repr):
        batch, freq_channels, frames, complex_dim = stft_repr.shape
        x = stft_repr.permute(0, 2, 1, 3).reshape(batch, frames, freq_channels * complex_dim)
        return self.model.band_split(x)


class MaskHeadWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        from pymss.modules.bs_roformer.common import mask_to_complex_shape

        return mask_to_complex_shape(self.model._estimate_masks(self.model.final_norm(x)), complex_dim=2)


def apply_rotary_export(cos, sin, t):
    cos = cos[..., ::2]
    sin = sin[..., ::2]
    even = t[..., ::2]
    odd = t[..., 1::2]
    return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(start_dim=-2)


def disable_grouped_forward(model: torch.nn.Module):
    previous = {"band_split": None, "mask_estimators": []}
    if hasattr(model, "band_split") and hasattr(model.band_split, "use_grouped_forward"):
        previous["band_split"] = model.band_split.use_grouped_forward
        model.band_split.use_grouped_forward = False
    for index, estimator in enumerate(getattr(model, "mask_estimators", [])):
        if hasattr(estimator, "use_grouped_forward"):
            previous["mask_estimators"].append((index, estimator.use_grouped_forward))
            estimator.use_grouped_forward = False
    return previous


def restore_grouped_forward(model: torch.nn.Module, previous) -> None:
    if previous["band_split"] is not None:
        model.band_split.use_grouped_forward = previous["band_split"]
    for index, value in previous["mask_estimators"]:
        model.mask_estimators[index].use_grouped_forward = value


def build_separator(preset: RoformerMNNPreset, *, batch_size: int = 1, device: str = "cpu") -> MSSeparator:
    logger = get_separation_logger()
    set_log_level(logger, 40)
    return MSSeparator(
        model_type=preset.model_type,
        model_path=str(preset.model_path),
        config_path=str(preset.config_path),
        device=device,
        device_ids=[0],
        output_format="wav",
        use_tta=False,
        store_dirs="",
        logger=logger,
        debug=False,
        inference_params={
            "batch_size": batch_size,
            "overlap_size": preset.overlap_size,
            "chunk_size": preset.chunk_size,
            "normalize": None,
            "mask_mode": preset.mask_mode,
        },
    )


def prepare_mask_core_model(separator: MSSeparator) -> MaskCoreWrapper:
    model = separator.model.cpu().eval()
    prepare_model_for_export(model)
    return MaskCoreWrapper(model).eval()


def prepare_model_for_export(model: torch.nn.Module) -> torch.nn.Module:
    model.float()
    for module in model.modules():
        if hasattr(module, "set_cuda_attention_backend"):
            module.set_cuda_attention_backend("default")
        if hasattr(module, "flash"):
            module.flash = False
        if hasattr(module, "attend") and hasattr(module.attend, "flash"):
            module.attend.flash = False
    disable_grouped_forward(model)
    model.eval()
    return model


def patch_rotary_for_export():
    previous = transformer_mod.apply_rotary_emb_fast
    transformer_mod.apply_rotary_emb_fast = apply_rotary_export
    return previous


def restore_rotary(previous) -> None:
    transformer_mod.apply_rotary_emb_fast = previous


def stft_input_from_audio(model: torch.nn.Module, audio: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        stft_repr, _ = stft_roformer(model, audio)
    return stft_repr.contiguous()


def random_stft_input(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float32)


def write_metadata(path: Path, preset: RoformerMNNPreset, separator: MSSeparator, output_shape: tuple[int, ...], *, compact: bool = True) -> None:
    model = separator.model.module if hasattr(separator.model, "module") else separator.model
    config = separator.config
    metadata = {
        "format": "pymss_roformer_mnn_mask_core",
        "version": 1,
        "preset": preset.name,
        "model_type": type(model).__name__,
        "model_path": str(preset.model_path.relative_to(ROOT)),
        "config_path": str(preset.config_path.relative_to(ROOT)),
        "input_name": "stft_repr",
        "output_name": "mask",
        "input_shape": list(preset.input_shape),
        "output_shape": list(output_shape),
        "mask_mode": str(getattr(model, "mask_mode", preset.mask_mode)),
        "sample_rate": int(config.audio.sample_rate),
        "chunk_size": int(config.audio.chunk_size),
        "overlap_size": int(config.inference.overlap_size),
        "source_names": list(preset.source_names),
        "stft": {
            "n_fft": int(model.stft_kwargs["n_fft"]),
            "hop_length": int(model.stft_kwargs["hop_length"]),
            "win_length": int(model.stft_kwargs["win_length"]),
            "normalized": bool(model.stft_kwargs["normalized"]),
            "window": "hann",
        },
        "mask_core_boundary": "input/output are real-imag split STFT tensors; audio DSP and overlap-add run outside MNN",
    }
    if hasattr(model, "freq_indices"):
        freq_indices = np.asarray(model.freq_indices.cpu(), dtype=np.int32)
        bands_per_freq = np.asarray(model.num_bands_per_channel_freq.cpu(), dtype=np.float32).reshape(-1)
        metadata["mel_band"] = {
            "freq_indices_shape": list(freq_indices.shape),
            "freq_indices_min": int(freq_indices.min()),
            "freq_indices_max": int(freq_indices.max()),
            "num_bands_per_channel_freq_shape": list(bands_per_freq.shape),
            "num_bands_per_channel_freq_min": float(bands_per_freq.min()),
            "num_bands_per_channel_freq_max": float(bands_per_freq.max()),
        }
        if not compact:
            metadata["mel_band"]["freq_indices"] = freq_indices.tolist()
            metadata["mel_band"]["num_bands_per_channel_freq"] = bands_per_freq.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def run_mnnconvert(onnx_path: Path, mnn_path: Path, *, timeout_seconds: int = 1800, extra_args: list[str] | None = None) -> str:
    cmd = [
        "MNNConvert",
        "-f",
        "ONNX",
        "--modelFile",
        str(onnx_path),
        "--MNNModel",
        str(mnn_path),
        "--bizCode",
        "pymss",
    ]
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_seconds)
    if proc.returncode != 0:
        raise RuntimeError(f"MNNConvert failed with code {proc.returncode}\n{proc.stdout}")
    return proc.stdout


def mnn_run_numpy(mnn_path: Path, input_name: str, output_name: str, x: np.ndarray, *, backend: str = "CPU", threads: int = 1) -> np.ndarray:
    import MNN

    runner = MNNRunner(mnn_path, input_name=input_name, output_name=output_name, backend=backend, threads=threads)
    return runner(x)


class MNNRunner:
    def __init__(self, mnn_path: Path | str, *, input_name: str = "input", output_name: str = "output", backend: str = "CPU", threads: int = 1):
        import MNN

        self.MNN = MNN
        self.interpreter = MNN.Interpreter(str(mnn_path))
        self.session = self.interpreter.createSession({"backend": backend, "numThread": threads})
        self.input_name = input_name
        self.output_name = output_name
        self._shape = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        MNN = self.MNN
        x = np.ascontiguousarray(x.astype(np.float32, copy=False))
        input_tensor = self.interpreter.getSessionInput(self.session, self.input_name)
        if self._shape != tuple(x.shape):
            self.interpreter.resizeTensor(input_tensor, tuple(x.shape))
            self.interpreter.resizeSession(self.session)
            self._shape = tuple(x.shape)
        host_input = MNN.Tensor(tuple(x.shape), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFromHostTensor(host_input)
        self.interpreter.runSession(self.session)
        output_tensor = self.interpreter.getSessionOutput(self.session, self.output_name)
        return np.asarray(output_tensor.getNumpyData(), dtype=np.float32).reshape(tuple(output_tensor.getShape()))


class MNNMultiRunner:
    def __init__(self, mnn_path: Path | str, *, input_names: tuple[str, ...], output_names: tuple[str, ...], backend: str = "CPU", threads: int = 1):
        import MNN

        self.MNN = MNN
        self.interpreter = MNN.Interpreter(str(mnn_path))
        self.session = self.interpreter.createSession({"backend": backend, "numThread": threads})
        self.input_names = tuple(input_names)
        self.output_names = tuple(output_names)
        self._shapes: dict[str, tuple[int, ...]] = {}

    def __call__(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        MNN = self.MNN
        for name in self.input_names:
            x = np.ascontiguousarray(inputs[name].astype(np.float32, copy=False))
            input_tensor = self.interpreter.getSessionInput(self.session, name)
            if self._shapes.get(name) != tuple(x.shape):
                self.interpreter.resizeTensor(input_tensor, tuple(x.shape))
                self._shapes[name] = tuple(x.shape)
            inputs[name] = x
        self.interpreter.resizeSession(self.session)
        for name in self.input_names:
            x = inputs[name]
            host_input = MNN.Tensor(tuple(x.shape), MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
            self.interpreter.getSessionInput(self.session, name).copyFromHostTensor(host_input)
        self.interpreter.runSession(self.session)
        outputs = {}
        for name in self.output_names:
            output_tensor = self.interpreter.getSessionOutput(self.session, name)
            outputs[name] = np.asarray(output_tensor.getNumpyData(), dtype=np.float32).reshape(tuple(output_tensor.getShape()))
        return outputs


def _mnn_run_numpy_uncached(mnn_path: Path, input_name: str, output_name: str, x: np.ndarray, *, backend: str = "CPU", threads: int = 1) -> np.ndarray:
    import MNN

    interpreter = MNN.Interpreter(str(mnn_path))
    session = interpreter.createSession({"backend": backend, "numThread": threads})
    input_tensor = interpreter.getSessionInput(session, input_name)
    interpreter.resizeTensor(input_tensor, tuple(x.shape))
    interpreter.resizeSession(session)
    x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    host_input = MNN.Tensor(tuple(x.shape), MNN.Halide_Type_Float, x.reshape(-1).tolist(), MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFromHostTensor(host_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session, output_name)
    return np.asarray(output_tensor.getNumpyData(), dtype=np.float32).reshape(tuple(output_tensor.getShape()))


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    diff = np.asarray(candidate, dtype=np.float32) - np.asarray(reference, dtype=np.float32)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "ref_rms": float(np.sqrt(np.mean(np.asarray(reference, dtype=np.float32) ** 2))),
    }
