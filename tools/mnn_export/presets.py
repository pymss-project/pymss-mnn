from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RoformerMNNPreset:
    name: str
    model_type: str
    model_path: Path
    config_path: Path
    chunk_size: int
    overlap_size: int
    input_shape: tuple[int, int, int, int]
    source_names: tuple[str, ...]
    mask_mode: str = "no_segm"


PRESETS = {
    "bsr_hyperace_voc": RoformerMNNPreset(
        name="bsr_hyperace_voc",
        model_type="bs_roformer",
        model_path=ROOT / "models/BS-Roformer-HyperACE_v2_voc/bs_roformer_voc_hyperacev2.ckpt",
        config_path=ROOT / "models/BS-Roformer-HyperACE_v2_voc/config.yaml",
        chunk_size=480000,
        overlap_size=24000,
        input_shape=(1, 2050, 938, 2),
        source_names=("vocals",),
    ),
    "bsr_hyperace_voc_full": RoformerMNNPreset(
        name="bsr_hyperace_voc_full",
        model_type="bs_roformer",
        model_path=ROOT / "models/BS-Roformer-HyperACE_v2_voc/bs_roformer_voc_hyperacev2.ckpt",
        config_path=ROOT / "models/BS-Roformer-HyperACE_v2_voc/config.yaml",
        chunk_size=480000,
        overlap_size=24000,
        input_shape=(1, 2050, 938, 2),
        source_names=("vocals",),
        mask_mode="full",
    ),
    "bsr_hyperace_voc_segm_only": RoformerMNNPreset(
        name="bsr_hyperace_voc_segm_only",
        model_type="bs_roformer",
        model_path=ROOT / "models/BS-Roformer-HyperACE_v2_voc/bs_roformer_voc_hyperacev2.ckpt",
        config_path=ROOT / "models/BS-Roformer-HyperACE_v2_voc/config.yaml",
        chunk_size=480000,
        overlap_size=24000,
        input_shape=(1, 2050, 938, 2),
        source_names=("vocals",),
        mask_mode="segm_only",
    ),
    "mbr_deux": RoformerMNNPreset(
        name="mbr_deux",
        model_type="mel_band_roformer",
        model_path=ROOT / "models/mel-band-roformer-deux/becruily_deux.ckpt",
        config_path=ROOT / "models/mel-band-roformer-deux/config_deux_becruily.yaml",
        chunk_size=573300,
        overlap_size=28665,
        input_shape=(1, 3958, 1301, 2),
        source_names=("Vocals", "Instrumental"),
    ),
    "logic_bs_roformer": RoformerMNNPreset(
        name="logic_bs_roformer",
        model_type="bs_roformer",
        model_path=ROOT / "models/logic_bs_roformer/logic_bs_roformer.ckpt",
        config_path=ROOT / "models/logic_bs_roformer/logic_pro_config.yaml",
        chunk_size=882000,
        overlap_size=44100,
        input_shape=(1, 2050, 1723, 2),
        source_names=("bass", "drums", "other", "vocals", "guitar", "piano"),
    ),
    "bs_roformer_ep_368": RoformerMNNPreset(
        name="bs_roformer_ep_368",
        model_type="bs_roformer",
        model_path=ROOT / "models/bs_roformer_ep_368/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        config_path=ROOT / "models/bs_roformer_ep_368/model_bs_roformer_ep_368_sdr_12.9628.yaml",
        chunk_size=352800,
        overlap_size=17640,
        input_shape=(1, 2050, 801, 2),
        source_names=("Vocals",),
    ),
    "mel_band_roformer_big": RoformerMNNPreset(
        name="mel_band_roformer_big",
        model_type="mel_band_roformer",
        model_path=ROOT / "models/Mel-Band-Roformer-big/big_beta7.ckpt",
        config_path=ROOT / "models/Mel-Band-Roformer-big/big_beta7.yaml",
        chunk_size=676935,
        overlap_size=338468,
        input_shape=(1, 3958, 1536, 2),
        source_names=("vocals",),
    ),
}


def preset_names() -> str:
    return ", ".join(sorted(PRESETS))


def get_preset(name: str) -> RoformerMNNPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; expected one of: {preset_names()}") from exc


def apply_shape_overrides(
    preset: RoformerMNNPreset,
    *,
    frames: int | None = None,
    chunk_size: int | None = None,
    overlap_size: int | None = None,
    variant_name: str | None = None,
) -> RoformerMNNPreset:
    if frames is None and chunk_size is None and overlap_size is None and variant_name is None:
        return preset
    base_frames = int(preset.input_shape[2])
    if base_frames <= 1:
        raise ValueError(f"cannot infer hop length from input_shape {preset.input_shape}")
    hop_length = max(1, int(round(preset.chunk_size / float(base_frames - 1))))
    resolved_frames = int(frames) if frames is not None else int(chunk_size) // hop_length + 1
    if resolved_frames <= 1:
        raise ValueError("--frames must be > 1")
    resolved_chunk_size = int(chunk_size) if chunk_size is not None else (resolved_frames - 1) * hop_length
    if resolved_chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")
    if overlap_size is not None:
        resolved_overlap_size = int(overlap_size)
    else:
        overlap_ratio = preset.overlap_size / float(max(1, preset.chunk_size))
        resolved_overlap_size = int(round(resolved_chunk_size * overlap_ratio))
    resolved_overlap_size = max(0, min(resolved_overlap_size, resolved_chunk_size - 1))
    batch, freq_channels, _, complex_dim = preset.input_shape
    resolved_variant_name = variant_name or f"{preset.name}_f{resolved_frames}"
    return replace(
        preset,
        name=resolved_variant_name,
        chunk_size=resolved_chunk_size,
        overlap_size=resolved_overlap_size,
        input_shape=(batch, freq_channels, resolved_frames, complex_dim),
    )


def default_out_dir() -> Path:
    return ROOT / "benchmark_results/mnn_work"
