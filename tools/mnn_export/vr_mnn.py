from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymss.audio_io import load_audio  # noqa: E402
from pymss.logger import get_separation_logger, set_log_level  # noqa: E402
from pymss.model_registry import config_path_for, get_model_entry, model_path_for  # noqa: E402
from pymss.modules.vocal_remover.common_separator import CommonSeparator  # noqa: E402
from pymss.modules.vocal_remover.uvr_lib_v5 import spec_utils  # noqa: E402
from pymss.modules.vocal_remover import VRSeparator  # noqa: E402
from pymss.separator import MSSeparator  # noqa: E402
from tools.mnn_export.presets import default_out_dir  # noqa: E402
from tools.mnn_export.roformer_mnn import metrics, run_mnnconvert  # noqa: E402


DOWNLOADED_VR_DIR = ROOT / "models/downloaded_vr"


@dataclass(frozen=True)
class VRMNNPreset:
    name: str
    model_name: str
    model_dir: Path
    input_shape: tuple[int, int, int, int]
    source_names: tuple[str, str]
    window_size: int = 512
    batch_size: int = 1


PRESETS = {
    "vr_denoise_lite": VRMNNPreset(
        name="vr_denoise_lite",
        model_name="UVR-DeNoise-Lite.pth",
        model_dir=DOWNLOADED_VR_DIR,
        input_shape=(1, 2, 1025, 512),
        source_names=("Noise", "No Noise"),
    ),
    "vr_mgm_main": VRMNNPreset(
        name="vr_mgm_main",
        model_name="MGM_MAIN_v4.pth",
        model_dir=DOWNLOADED_VR_DIR,
        input_shape=(1, 2, 1025, 512),
        source_names=("Instrumental", "Vocals"),
    ),
}


def vr_preset_names() -> str:
    return ", ".join(sorted(PRESETS))


def get_vr_preset(name: str) -> VRMNNPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown VR preset {name!r}; expected one of: {vr_preset_names()}") from exc


def build_vr_separator(preset: VRMNNPreset, *, device: str = "cpu") -> MSSeparator:
    entry = get_model_entry(preset.model_name)
    logger = get_separation_logger()
    set_log_level(logger, 40)
    return MSSeparator(
        model_type="vr",
        model_path=str(model_path_for(entry, preset.model_dir)),
        config_path=str(config_path_for(entry, preset.model_dir)),
        device=device,
        device_ids=[0],
        output_format="wav",
        use_tta=False,
        store_dirs="",
        logger=logger,
        debug=False,
        inference_params={
            "batch_size": preset.batch_size,
            "window_size": preset.window_size,
            "aggression": 5,
            "enable_tta": False,
            "enable_post_process": False,
            "high_end_process": False,
            "use_amp": False,
            "fuse_conv_bn": False,
            "use_channels_last": False,
        },
    )


class VRPredictMaskWrapper(torch.nn.Module):
    def __init__(self, model_run: torch.nn.Module):
        super().__init__()
        self.model_run = model_run.cpu().float().eval()

    def forward(self, mag_patch):
        return self.model_run.predict_mask(mag_patch.float())


def prepare_vr_core_model(separator: MSSeparator) -> VRPredictMaskWrapper:
    model = separator.model
    if not isinstance(model, VRSeparator):
        raise TypeError(f"expected VRSeparator, got {type(model).__name__}")
    if model.model_run is None:
        model.load_model()
    model.model_run.cpu().float().eval()
    return VRPredictMaskWrapper(model.model_run).eval()


def random_vr_input(preset: VRMNNPreset, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(*preset.input_shape, generator=generator, dtype=torch.float32)


def write_vr_metadata(path: Path, preset: VRMNNPreset, separator: MSSeparator, output_shape: tuple[int, ...]) -> None:
    model = separator.model
    entry = get_model_entry(preset.model_name)
    metadata = {
        "format": "pymss_vr_mnn_core",
        "version": 1,
        "preset": preset.name,
        "model_name": preset.model_name,
        "model_path": str(model_path_for(entry, preset.model_dir).relative_to(ROOT)),
        "config_path": str(config_path_for(entry, preset.model_dir).relative_to(ROOT)),
        "input_name": "mag_patch",
        "output_name": "mask_patch",
        "input_shape": list(preset.input_shape),
        "output_shape": list(output_shape),
        "batch_size": int(preset.batch_size),
        "window_size": int(preset.window_size),
        "source_names": list(preset.source_names),
        "model_samplerate": int(model.model_samplerate),
        "is_vr_51_model": bool(model.is_vr_51_model),
        "model_capacity": list(model.model_capacity),
        "offset": int(model.model_run.offset),
        "bins": int(model.model_params.param["bins"]),
        "core_boundary": "VR STFT/ISTFT, padding, aggressiveness and stem reconstruction run outside MNN; predict_mask neural core runs in MNN.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_vr_audio(path: Path, sample_rate: int, seconds: float | None = None) -> tuple[np.ndarray, int]:
    audio, sr = load_audio(str(path), sr=sample_rate, mono=False, duration=seconds)
    return np.asarray(audio, dtype=np.float32), int(sr)


def apply_vr_aggressiveness(mask, separator: VRSeparator):
    aggr = separator.aggressiveness["value"] * 2
    is_non_accom_stem = separator.primary_stem_name in CommonSeparator.NON_ACCOM_STEMS
    if is_non_accom_stem:
        aggr = 1 - aggr

    aggr_arr = np.array([aggr, aggr], dtype=np.float32)
    if (correction := separator.aggressiveness["aggr_correction"]) is not None:
        aggr_arr[0] += correction["left"]
        aggr_arr[1] += correction["right"]

    split_bin = int(separator.aggressiveness["split_bin"])
    out = np.asarray(mask, dtype=np.float32).copy()
    out[:, :split_bin] = np.power(out[:, :split_bin], 1 + aggr_arr[:, None, None] / 3)
    out[:, split_bin:] = np.power(out[:, split_bin:], 1 + aggr_arr[:, None, None])
    return out


def vr_mask_from_runner(separator: VRSeparator, x_spec: np.ndarray, runner) -> np.ndarray:
    x_mag = np.abs(x_spec).astype(np.float32, copy=False)
    n_frame = x_mag.shape[2]
    pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, separator.window_size, separator.model_run.offset)
    x_mag_pad = np.pad(x_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
    max_value = float(x_mag_pad.max())
    if max_value > 0:
        x_mag_pad /= max_value

    patches = (x_mag_pad.shape[2] - 2 * separator.model_run.offset) // roi_size
    if patches <= 0:
        raise ValueError("Window size error: no VR patches generated")

    mask_chunks = []
    for start in range(0, patches, int(separator.batch_size)):
        batch = np.asarray(
            [
                x_mag_pad[:, :, i * roi_size : i * roi_size + separator.window_size]
                for i in range(start, min(start + int(separator.batch_size), patches))
            ],
            dtype=np.float32,
        )
        pred = runner(batch)
        if pred.shape[3] <= 0:
            raise ValueError("Window size error: h1_shape[3] must be greater than h2_shape[3]")
        mask_chunks.append(pred.transpose(1, 2, 0, 3).reshape(pred.shape[1], pred.shape[2], -1))

    mask = np.concatenate(mask_chunks, axis=2)[:, :, :n_frame]
    return apply_vr_aggressiveness(mask, separator)


def vr_outputs_from_mask(separator: VRSeparator, x_spec: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    y_spec = np.nan_to_num(mask * x_spec, nan=0.0, posinf=0.0, neginf=0.0)
    v_spec = np.nan_to_num((1 - mask) * x_spec, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        separator.primary_stem_name: separator.process_stem(None, y_spec),
        separator.secondary_stem_name: separator.process_stem(None, v_spec),
    }


def separate_vr_with_runner(separator: VRSeparator, mix: np.ndarray, sample_rate: int, runner) -> dict[str, np.ndarray]:
    x_spec = separator.loading_mix(mix, sample_rate)
    mask = vr_mask_from_runner(separator, x_spec, runner)
    return vr_outputs_from_mask(separator, x_spec, mask)


def vr_torch_runner(separator: VRSeparator):
    def run(batch_np: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            batch = torch.from_numpy(np.ascontiguousarray(batch_np)).float()
            return separator.model_run.predict_mask(batch).detach().cpu().numpy().astype(np.float32, copy=False)

    return run


def audio_metrics_by_stem(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    result = {}
    for name in reference:
        if name not in candidate:
            continue
        ref = np.asarray(reference[name], dtype=np.float32)
        cand = np.asarray(candidate[name], dtype=np.float32)
        length = min(ref.shape[-1], cand.shape[-1])
        result[name] = metrics(ref[..., :length], cand[..., :length])
    return result


__all__ = [
    "PRESETS",
    "VRMNNPreset",
    "audio_metrics_by_stem",
    "build_vr_separator",
    "default_out_dir",
    "get_vr_preset",
    "load_vr_audio",
    "metrics",
    "prepare_vr_core_model",
    "random_vr_input",
    "run_mnnconvert",
    "separate_vr_with_runner",
    "vr_preset_names",
    "vr_torch_runner",
    "write_vr_metadata",
]
