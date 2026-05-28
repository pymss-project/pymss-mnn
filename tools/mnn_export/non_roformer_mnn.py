from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.mnn_export.presets import default_out_dir  # noqa: E402
from tools.mnn_export.roformer_mnn import build_separator, metrics, run_mnnconvert  # noqa: E402
from pymss.modules.spectrogram import cac_to_cws, cws_to_cac  # noqa: E402


@dataclass(frozen=True)
class NonRoformerMNNPreset:
    name: str
    model_type: str
    model_path: Path
    config_path: Path
    chunk_size: int
    overlap_size: int
    source_names: tuple[str, ...]
    core_kind: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...] = ()
    mask_mode: str = "no_segm"


PRESETS = {
    "scnet_similarity": NonRoformerMNNPreset(
        name="scnet_similarity",
        model_type="scnet",
        model_path=ROOT / "models/model_scnet_ep_102_sdr_12.8941.ckpt",
        config_path=ROOT / "models/config_scnet_similarity.yaml",
        chunk_size=131072,
        overlap_size=0,
        source_names=("similarity",),
        core_kind="scnet_core",
        input_shapes=((1, 4, 2049, 130),),
        input_names=("stft_repr",),
        output_names=("stft_out",),
    ),
    "scnet_difference": NonRoformerMNNPreset(
        name="scnet_difference",
        model_type="scnet",
        model_path=ROOT / "models/model_scnet_ep_30_sdr_15.1291.ckpt",
        config_path=ROOT / "models/config_scnet_difference.yaml",
        chunk_size=131072,
        overlap_size=0,
        source_names=("difference",),
        core_kind="scnet_core",
        input_shapes=((1, 4, 2049, 130),),
        input_names=("stft_repr",),
        output_names=("stft_out",),
    ),
    "htdemucs_similarity": NonRoformerMNNPreset(
        name="htdemucs_similarity",
        model_type="htdemucs",
        model_path=ROOT / "models/model_htdemucs_ep_21_sdr_13.6970.ckpt",
        config_path=ROOT / "models/config_htdemucs_similarity.yaml",
        chunk_size=132300,
        overlap_size=0,
        source_names=("similarity", "difference"),
        core_kind="htdemucs_core",
        input_shapes=((1, 4, 2048, 130), (1, 2, 132300)),
        input_names=("mag", "mix"),
        output_names=("mask", "time"),
    ),
    "smoke_scnet": NonRoformerMNNPreset(
        name="smoke_scnet",
        model_type="scnet",
        model_path=ROOT / "models/smoke/scnet/scnet_checkpoint_musdb18.ckpt",
        config_path=ROOT / "models/smoke/scnet/config.yaml",
        chunk_size=485100,
        overlap_size=0,
        source_names=("drums", "bass", "other", "vocals"),
        core_kind="scnet_core",
        input_shapes=((1, 4, 2049, 476),),
        input_names=("stft_repr",),
        output_names=("stft_out",),
    ),
    "smoke_htdemucs": NonRoformerMNNPreset(
        name="smoke_htdemucs",
        model_type="htdemucs",
        model_path=ROOT / "models/smoke/htdemucs/HTDemucs4.th",
        config_path=ROOT / "models/smoke/htdemucs/config.yaml",
        chunk_size=485100,
        overlap_size=0,
        source_names=("drums", "bass", "other", "vocals"),
        core_kind="htdemucs_core",
        input_shapes=((1, 4, 2048, 474), (1, 2, 485100)),
        input_names=("mag", "mix"),
        output_names=("mask", "time"),
    ),
    "smoke_mdx23c": NonRoformerMNNPreset(
        name="smoke_mdx23c",
        model_type="mdx23c",
        model_path=ROOT / "models/smoke/mdx23c/model_vocals_mdx23c_sdr_10.17.ckpt",
        config_path=ROOT / "models/smoke/mdx23c/config.yaml",
        chunk_size=261120,
        overlap_size=0,
        source_names=("vocals", "other"),
        core_kind="mdx23c_core",
        input_shapes=((1, 16, 1024, 256),),
        input_names=("cws_spec",),
        output_names=("cac_spec",),
    ),
    "smoke_apollo": NonRoformerMNNPreset(
        name="smoke_apollo",
        model_type="apollo",
        model_path=ROOT / "models/smoke/apollo/Apollo_LQ_MP3_restoration.ckpt",
        config_path=ROOT / "models/smoke/apollo/config.yaml",
        chunk_size=132300,
        overlap_size=0,
        source_names=("restored",),
        core_kind="apollo_core",
        input_shapes=((2, 2, 442, 301),),
        input_names=("stft_ri",),
        output_names=("est_spec_ri",),
    ),
    "smoke_bandit": NonRoformerMNNPreset(
        name="smoke_bandit",
        model_type="bandit",
        model_path=ROOT / "models/smoke/bandit/model_bandit_plus_dnr_sdr_11.47.chpt",
        config_path=ROOT / "models/smoke/bandit/config.yaml",
        chunk_size=264600,
        overlap_size=0,
        source_names=("speech", "music", "effects"),
        core_kind="bandit_core",
        input_shapes=((1, 2, 2, 1025, 517),),
        input_names=("stft_ri",),
        output_names=("est_spec_ri",),
    ),
    "smoke_bandit_v2": NonRoformerMNNPreset(
        name="smoke_bandit_v2",
        model_type="bandit_v2",
        model_path=ROOT / "models/smoke/bandit_v2/checkpoint-multi_state_dict.ckpt",
        config_path=ROOT / "models/smoke/bandit_v2/config.yaml",
        chunk_size=384000,
        overlap_size=0,
        source_names=("speech", "music", "sfx"),
        core_kind="bandit_v2_core",
        input_shapes=((2, 1, 2, 1025, 751),),
        input_names=("stft_ri",),
        output_names=("est_spec_ri",),
    ),
}


def non_roformer_preset_names() -> str:
    return ", ".join(sorted(PRESETS))


def get_non_roformer_preset(name: str) -> NonRoformerMNNPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"unknown preset {name!r}; expected one of: {non_roformer_preset_names()}") from exc


def build_non_roformer_separator(preset: NonRoformerMNNPreset, *, batch_size: int = 1, device: str = "cpu"):
    separator = build_separator(preset, batch_size=batch_size, device=device)
    separator.config.inference.batch_size = 1
    separator.config.inference.overlap_size = int(preset.overlap_size)
    return separator


def _rfft_matrices(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    basis = torch.eye(length, dtype=torch.float32)
    freq = torch.fft.rfft(basis, dim=0, norm="ortho").transpose(0, 1).contiguous()
    return freq.real.contiguous(), freq.imag.contiguous()


def _irfft_matrices(freq_bins: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    real_rows = []
    imag_rows = []
    for index in range(freq_bins):
        real_spec = torch.zeros(freq_bins, dtype=torch.complex64)
        real_spec[index] = 1
        real_rows.append(torch.fft.irfft(real_spec, n=length, norm="ortho"))
        imag_spec = torch.zeros(freq_bins, dtype=torch.complex64)
        imag_spec[index] = 1j
        imag_rows.append(torch.fft.irfft(imag_spec, n=length, norm="ortho"))
    return torch.stack(real_rows, dim=0).contiguous(), torch.stack(imag_rows, dim=0).contiguous()


class FixedFeatureConversion(torch.nn.Module):
    def __init__(self, channels: int, inverse: bool, time_length: int):
        super().__init__()
        self.inverse = inverse
        self.channels = channels
        if inverse:
            half = channels // 2
            freq_bins = time_length
            length = (freq_bins - 1) * 2
            real, imag = _irfft_matrices(freq_bins, length)
            self.output_channels = half
        else:
            real, imag = _rfft_matrices(time_length)
            self.output_channels = channels * 2
        self.register_buffer("real_matrix", real)
        self.register_buffer("imag_matrix", imag)

    def forward(self, x):
        x = x.float()
        if self.inverse:
            half = self.channels // 2
            real = x[:, :half]
            imag = x[:, half:]
            return torch.matmul(real, self.real_matrix) + torch.matmul(imag, self.imag_matrix)
        real = torch.matmul(x, self.real_matrix)
        imag = torch.matmul(x, self.imag_matrix)
        return torch.cat([real, imag], dim=1)


def patch_scnet_fft_for_export(model: torch.nn.Module, *, time_length: int) -> None:
    for index, module in enumerate(model.separation_net.feature_conversion):
        model.separation_net.feature_conversion[index] = FixedFeatureConversion(
            int(module.channels),
            bool(module.inverse),
            time_length if not module.inverse else time_length // 2 + 1,
        )


class SCNetCoreWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, patch_fft: bool = True, time_length: int = 130):
        super().__init__()
        self.model = model.float().cpu().eval()
        if patch_fft:
            patch_scnet_fft_for_export(self.model, time_length=time_length)

    def forward(self, stft_repr):
        x = stft_repr
        saved = deque()
        for sd_layer in self.model.encoder:
            x, skip, lengths, original_lengths = sd_layer(x)
            saved.append((skip, lengths, original_lengths))

        x = self.model.separation_net(x)

        for fusion_layer, su_layer in self.model.decoder:
            skip, lengths, original_lengths = saved.pop()
            x = su_layer(fusion_layer(x, skip), lengths, original_lengths)
        return x


def scnet_stft_input(model: torch.nn.Module, audio: torch.Tensor) -> tuple[torch.Tensor, int]:
    padding = model.hop_length - audio.shape[-1] % model.hop_length
    if (audio.shape[-1] + padding) // model.hop_length % 2 == 0:
        padding += model.hop_length
    padded = F.pad(audio, (0, padding))
    length = padded.shape[-1]
    spec = torch.stft(padded.reshape(-1, length), **model.stft_config, return_complex=True)
    stft_repr = torch.view_as_real(spec)
    stft_repr = stft_repr.permute(0, 3, 1, 2)
    stft_repr = stft_repr.reshape(stft_repr.shape[0] // model.audio_channels, stft_repr.shape[1] * model.audio_channels, stft_repr.shape[2], stft_repr.shape[3])
    return stft_repr.contiguous(), int(padding)


def scnet_istft_output(model: torch.nn.Module, stft_out: torch.Tensor, *, padding: int) -> torch.Tensor:
    batch, _, freq_bins, frames = stft_out.shape
    channels = int(model.dims[0])
    spec = stft_out.view(batch, channels, -1, freq_bins, frames)
    spec = spec.reshape(-1, 2, freq_bins, frames).permute(0, 2, 3, 1).contiguous()
    audio = torch.istft(torch.view_as_complex(spec), **model.stft_config)
    audio = audio.reshape(batch, len(model.sources), model.audio_channels, -1)
    return audio[:, :, :, :-padding] if padding > 0 else audio


class HTDemucsCoreWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.float().cpu().eval()

    def forward(self, mag, mix):
        model = self.model
        x = mag
        if model.num_subbands > 1:
            x = model.cac2cws(x)

        batch, _, freq_bins, frames = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        saved, saved_t, lengths_t = [], [], []
        for idx, encode in enumerate(model.encoder):
            skip_length = x.shape[-1]
            inject = None
            if idx < len(model.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = model.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and model.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = model.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + model.freq_emb_scale * emb
            saved.append((x, skip_length))

        if model.crosstransformer:
            if model.bottom_channels:
                b, c, f, t = x.shape
                x = model.channel_upsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = model.channel_upsampler_t(xt)

            x, xt = model.crosstransformer(x, xt)

            if model.bottom_channels:
                b, c, f, t = x.shape
                x = model.channel_downsampler(x.reshape(b, c, f * t)).reshape(b, -1, f, t)
                xt = model.channel_downsampler_t(xt)

        for idx, decode in enumerate(model.decoder):
            skip, skip_length = saved.pop(-1)
            x, pre = decode(x, skip, skip_length)

            offset = model.depth - len(model.tdecoder)
            if idx >= offset:
                tdec = model.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        sources = len(model.sources)
        if model.num_subbands > 1:
            x = model.cws2cac(x.view(batch, -1, freq_bins, frames))

        mask = x.view(batch, sources, -1, freq_bins * model.num_subbands, frames)
        mask = mask * std[:, None] + mean[:, None]
        time = xt.view(batch, sources, -1, mix.shape[-1])
        time = time * stdt[:, None] + meant[:, None]
        return mask, time


def htdemucs_core_inputs(model: torch.nn.Module, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = model._spec(audio)
    return model._magnitude(z).contiguous(), audio.contiguous(), z


def htdemucs_finish_output(model: torch.nn.Module, z: torch.Tensor, mask: torch.Tensor, time: torch.Tensor, *, length: int) -> torch.Tensor:
    spec_audio = model._ispec(model._mask(z, mask), length)
    return time + spec_audio


class MDX23CCoreWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.float().cpu().eval()

    def forward(self, cws_spec):
        first_conv_out = self.model.first_conv(cws_spec)
        core = self.model._forward_core(first_conv_out.transpose(-1, -2)).transpose(-1, -2)
        masked = self.model.final_conv(torch.cat([cws_spec, first_conv_out * core], 1))
        return cws_to_cac(masked, self.model.num_subbands)


def mdx23c_core_input(model: torch.nn.Module, audio: torch.Tensor) -> torch.Tensor:
    return cac_to_cws(model.stft(audio), model.num_subbands).contiguous()


def mdx23c_finish_output(model: torch.nn.Module, cac_spec: torch.Tensor) -> torch.Tensor:
    if model.num_target_instruments > 1:
        batch, channels, freq_bins, time_bins = cac_spec.shape
        cac_spec = cac_spec.reshape(batch, model.num_target_instruments, -1, freq_bins, time_bins)
    return model.stft.inverse(cac_spec)


class ApolloCoreWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.float().cpu().eval()

    def forward(self, stft_ri):
        spec_real = stft_ri[:, 0]
        spec_imag = stft_ri[:, 1]
        subband_features, band_idx = [], 0
        for index, width in enumerate(self.model.band_width):
            real = spec_real[:, band_idx : band_idx + width]
            imag = spec_imag[:, band_idx : band_idx + width]
            power = (real.pow(2).sum(1) + imag.pow(2).sum(1) + self.model.eps).sqrt()
            feature = torch.cat([real / power.unsqueeze(1), imag / power.unsqueeze(1), torch.log(power).unsqueeze(1)], 1)
            subband_features.append(self.model.BN[index](feature))
            band_idx += width

        feature = self.model.net(torch.stack(subband_features, 1))
        outputs = []
        for index, width in enumerate(self.model.band_width):
            outputs.append(self.model.output[index](feature[:, index]).view(stft_ri.shape[0], 2, width, -1))
        return torch.cat(outputs, 2)


def apollo_core_input(model: torch.nn.Module, audio: torch.Tensor) -> torch.Tensor:
    batch, channels, samples = audio.shape
    spec = torch.stft(
        audio.view(batch * channels, samples),
        n_fft=model.win,
        hop_length=model.stride,
        window=model._window(audio),
        return_complex=True,
    )
    return torch.stack([spec.real, spec.imag], dim=1).contiguous()


def apollo_finish_output(model: torch.nn.Module, est_spec_ri: torch.Tensor, *, batch: int, channels: int, length: int) -> torch.Tensor:
    spec = torch.complex(est_spec_ri[:, 0].float(), est_spec_ri[:, 1].float())
    audio = torch.istft(
        spec,
        n_fft=model.win,
        hop_length=model.stride,
        window=model._window(est_spec_ri),
        length=length,
    )
    return audio.view(batch, channels, -1)


def _ri_to_complex_spec(stft_ri: torch.Tensor) -> torch.Tensor:
    return torch.complex(stft_ri[..., 0, :, :].float(), stft_ri[..., 1, :, :].float())


def _complex_spec_to_ri(spec: torch.Tensor) -> torch.Tensor:
    return torch.stack([spec.real, spec.imag], dim=-3)


def _bandit_real_band_split(band_split: torch.nn.Module, x_ri: torch.Tensor) -> torch.Tensor:
    batch, in_channels, _, _, n_time = x_ri.shape
    outputs = []
    if band_split.complex_order == "reim_freq":
        xr = x_ri.permute(0, 4, 1, 2, 3)
    elif band_split.complex_order == "freq_reim":
        xr = x_ri.permute(0, 4, 1, 3, 2).contiguous()
    else:
        raise ValueError(f"unsupported Bandit complex_order: {band_split.complex_order}")

    for index, nfm in enumerate(band_split.norm_fc_modules):
        fstart, fend = band_split.band_specs[index]
        if band_split.complex_order == "reim_freq":
            xb = xr[..., fstart:fend].reshape(batch, n_time, in_channels, -1)
        else:
            xb = xr[:, :, :, fstart:fend].reshape(batch, n_time, -1)
        xb = (xb.reshape(batch, n_time, -1) if band_split.flatten_input else xb).contiguous()
        outputs.append(nfm.combined(xb) if hasattr(nfm, "combined") else nfm(xb))
    return torch.stack(outputs, dim=1)


def _bandit_norm_mlp_ri(norm_mlp: torch.nn.Module, qb: torch.Tensor) -> torch.Tensor:
    if hasattr(norm_mlp, "combined"):
        mb = norm_mlp.combined(qb)
    else:
        mb = norm_mlp.output(norm_mlp.hidden(norm_mlp.norm(qb)))

    batch, n_time, _ = mb.shape
    if norm_mlp.complex_mask:
        return mb.reshape(batch, n_time, norm_mlp.in_channels, norm_mlp.bandwidth, 2).permute(0, 2, 4, 3, 1)

    real = mb.reshape(batch, n_time, norm_mlp.in_channels, norm_mlp.bandwidth).permute(0, 2, 3, 1)
    imag = torch.zeros_like(real)
    return torch.stack([real, imag], dim=2)


def _bandit_mask_estimator_ri(mask_estimator: torch.nn.Module, q: torch.Tensor, cond=None) -> torch.Tensor:
    q = mask_estimator._append_cond(q, cond)
    batch, _, n_time, _ = q.shape

    if getattr(mask_estimator, "n_freq", 0):
        masks = torch.zeros(
            batch,
            mask_estimator.in_channels,
            2,
            mask_estimator.n_freq,
            n_time,
            device=q.device,
            dtype=q.dtype,
        )
        for index, norm_mlp in enumerate(mask_estimator.norm_mlp):
            fstart, fend = mask_estimator.band_specs[index]
            mask = _bandit_norm_mlp_ri(norm_mlp, q[:, index, :, :])
            if mask_estimator.use_freq_weights:
                mask = mask * mask_estimator.get_buffer(f"freq_weights/{index}")[None, None, None, :, None]
            masks[:, :, :, fstart:fend, :] += mask
        return masks

    return torch.cat(
        [_bandit_norm_mlp_ri(norm_mlp, q[:, index, :, :]) for index, norm_mlp in enumerate(mask_estimator.norm_mlp)],
        dim=3,
    )


def _complex_mul_ri(left_ri: torch.Tensor, right_ri: torch.Tensor) -> torch.Tensor:
    left_real, left_imag = left_ri[:, :, 0], left_ri[:, :, 1]
    right_real, right_imag = right_ri[:, :, 0], right_ri[:, :, 1]
    return torch.stack(
        [
            left_real * right_real - left_imag * right_imag,
            left_real * right_imag + left_imag * right_real,
        ],
        dim=2,
    )


class BanditCoreWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.float().cpu().eval()

    def forward(self, stft_ri):
        batch, in_channels, _, n_freq, n_time = stft_ri.shape
        x = stft_ri.float().reshape(batch * in_channels, 1, 2, n_freq, n_time)
        q = self.model.bsrnn.tf_model(_bandit_real_band_split(self.model.bsrnn.band_split, x))
        estimates = []
        for stem in self.model.stems:
            mask = _bandit_mask_estimator_ri(self.model.bsrnn.mask_estim[stem], q)
            estimates.append(_complex_mul_ri(x, mask).reshape(batch, in_channels, 2, n_freq, n_time))
        return torch.stack(estimates, dim=1)


class BanditV2CoreWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model.float().cpu().eval()

    def forward(self, stft_ri):
        x = stft_ri.float()
        q = self.model.tf_model(_bandit_real_band_split(self.model.band_split, x))
        return torch.stack(
            [
                _complex_mul_ri(x, _bandit_mask_estimator_ri(self.model.mask_estim[stem], q).to(x.dtype)).reshape(x.shape)
                for stem in self.model.stems
            ],
            dim=1,
        )


def bandit_core_input(model: torch.nn.Module, audio: torch.Tensor, *, flatten_channels: bool = False) -> torch.Tensor:
    source = audio.view(-1, 1, audio.shape[-1]) if flatten_channels else audio
    spec = model.stft(source)
    return _complex_spec_to_ri(spec).contiguous()


def bandit_finish_output(model: torch.nn.Module, est_spec_ri: torch.Tensor, *, length: int, batch: int | None = None, channels: int | None = None) -> torch.Tensor:
    stems = []
    for index in range(est_spec_ri.shape[1]):
        audio = model.istft(_ri_to_complex_spec(est_spec_ri[:, index]), length)
        if batch is not None and channels is not None:
            audio = audio.view(batch, channels, length)
        stems.append(audio)
    return torch.stack(stems, dim=1)


def prepare_non_roformer_core_model(separator, preset: NonRoformerMNNPreset) -> torch.nn.Module:
    model = separator.model.cpu().float().eval()
    if preset.core_kind == "scnet_core":
        return SCNetCoreWrapper(model, patch_fft=True, time_length=preset.input_shapes[0][-1]).eval()
    if preset.core_kind == "htdemucs_core":
        return HTDemucsCoreWrapper(model).eval()
    if preset.core_kind == "mdx23c_core":
        return MDX23CCoreWrapper(model).eval()
    if preset.core_kind == "apollo_core":
        return ApolloCoreWrapper(model).eval()
    if preset.core_kind == "bandit_core":
        return BanditCoreWrapper(model).eval()
    if preset.core_kind == "bandit_v2_core":
        return BanditV2CoreWrapper(model).eval()
    raise ValueError(f"unsupported non-RoFormer core kind: {preset.core_kind}")


def random_inputs(preset: NonRoformerMNNPreset, seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return tuple(torch.randn(*shape, generator=generator, dtype=torch.float32) for shape in preset.input_shapes)


def write_non_roformer_metadata(path: Path, preset: NonRoformerMNNPreset, separator, output_shapes: tuple[tuple[int, ...], ...]) -> None:
    model = separator.model.module if hasattr(separator.model, "module") else separator.model
    metadata = {
        "format": "pymss_non_roformer_mnn_core",
        "version": 1,
        "preset": preset.name,
        "core_kind": preset.core_kind,
        "model_type": type(model).__name__,
        "model_path": str(preset.model_path.relative_to(ROOT)),
        "config_path": str(preset.config_path.relative_to(ROOT)),
        "input_names": list(preset.input_names),
        "output_names": list(preset.output_names),
        "input_shapes": [list(shape) for shape in preset.input_shapes],
        "output_shapes": [list(shape) for shape in output_shapes],
        "sample_rate": int(separator.config.audio.get("sample_rate", separator.config.training.get("samplerate", 44100))),
        "chunk_size": int(preset.chunk_size),
        "batch_size": 1,
        "overlap_size": int(preset.overlap_size),
        "source_names": list(preset.source_names),
        "core_boundary": "audio STFT/ISTFT and chunk overlap-add run outside MNN; neural network core runs in MNN",
    }
    if preset.core_kind == "scnet_core":
        metadata["stft"] = {
            "n_fft": int(model.stft_config["n_fft"]),
            "hop_length": int(model.stft_config["hop_length"]),
            "win_length": int(model.stft_config["win_length"]),
            "normalized": bool(model.stft_config["normalized"]),
            "center": bool(model.stft_config["center"]),
            "window": "default_torch_rectangular" if "window" not in model.stft_config else "configured",
        }
        metadata["fixed_fft"] = "SCNet internal rfft/irfft feature conversions are exported as real-valued constant matrix multiplications."
    if preset.core_kind == "htdemucs_core":
        metadata["stft"] = {
            "n_fft": int(model.nfft),
            "hop_length": int(model.hop_length),
            "normalized": True,
            "center": True,
            "window": "hann",
            "frequency_bins_without_nyquist": int(model.nfft // 2),
        }
    if preset.core_kind == "mdx23c_core":
        metadata["stft"] = {
            "n_fft": int(model.stft.n_fft),
            "hop_length": int(model.stft.hop_length),
            "dim_f": int(model.stft.dim_f),
            "window": "hann",
        }
    if preset.core_kind == "apollo_core":
        metadata["stft"] = {
            "n_fft": int(model.win),
            "hop_length": int(model.stride),
            "frequency_bins": int(model.enc_dim),
            "window": "hann",
        }
    if preset.core_kind in {"bandit_core", "bandit_v2_core"}:
        metadata["stft"] = {
            "n_fft": int(model.stft.n_fft),
            "hop_length": int(model.stft.hop_length),
            "win_length": int(model.stft.win_length),
            "normalized": bool(model.stft.normalized),
            "center": bool(model.stft.center),
            "onesided": bool(model.stft.onesided),
            "window": "hann",
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "NonRoformerMNNPreset",
    "apollo_core_input",
    "apollo_finish_output",
    "bandit_core_input",
    "bandit_finish_output",
    "PRESETS",
    "build_non_roformer_separator",
    "default_out_dir",
    "get_non_roformer_preset",
    "htdemucs_core_inputs",
    "htdemucs_finish_output",
    "mdx23c_core_input",
    "mdx23c_finish_output",
    "metrics",
    "non_roformer_preset_names",
    "prepare_non_roformer_core_model",
    "random_inputs",
    "run_mnnconvert",
    "scnet_istft_output",
    "scnet_stft_input",
    "write_non_roformer_metadata",
]
