import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from tqdm import tqdm

from .common_separator import CommonSeparator
from .uvr_lib_v5 import spec_utils
from .uvr_lib_v5.vr_network import nets, nets_new
from .uvr_lib_v5.vr_network.model_param_init import ModelParameters


VR_PARAMS_DIR = Path(__file__).resolve().parents[2] / "resources" / "vr_modelparams"


def _fuse_sequential_conv_bn(module):
    fused = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            new_children = []
            child_items = list(child._modules.items())
            i = 0
            while i < len(child_items):
                child_name, current = child_items[i]
                if (
                    i + 1 < len(child_items)
                    and isinstance(current, nn.Conv2d)
                    and isinstance(child_items[i + 1][1], nn.BatchNorm2d)
                ):
                    try:
                        new_children.append((child_name, fuse_conv_bn_eval(current, child_items[i + 1][1])))
                        fused += 1
                        i += 2
                        continue
                    except Exception:
                        pass
                child_fused = _fuse_sequential_conv_bn(current)
                fused += child_fused
                new_children.append((child_name, current))
                i += 1

            child._modules.clear()
            for child_name, current in new_children:
                child.add_module(child_name, current)
        else:
            fused += _fuse_sequential_conv_bn(child)
    return fused


class VRSeparator(CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(common_config)
        self.model_capacity = (32, 128)
        self.is_vr_51_model = False
        if "nout" in self.model_data and "nout_lstm" in self.model_data:
            self.model_capacity = (self.model_data["nout"], self.model_data["nout_lstm"])
            self.is_vr_51_model = True

        params_path = VR_PARAMS_DIR / f"{self.model_data['vr_model_param']}.json"
        if not params_path.exists():
            raise FileNotFoundError(f"VR model parameter file not found: {params_path}")
        self.model_params = ModelParameters(str(params_path))

        self.batch_size = int(arch_config.get("batch_size", 2))
        self.window_size = int(arch_config.get("window_size", 512))
        self.high_end_process = bool(arch_config.get("high_end_process", False))
        device_type = torch.device(self.torch_device).type
        self.fuse_conv_bn = bool(arch_config.get("fuse_conv_bn", False))
        self.use_channels_last = bool(arch_config.get("use_channels_last", False)) and device_type == "cuda"
        self.input_high_end_h = None
        self.input_high_end = None
        self.aggression = float(int(arch_config.get("aggression", 5)) / 100)
        self.aggressiveness = {
            "value": self.aggression,
            "split_bin": self.model_params.param["band"][1]["crop_stop"],
            "aggr_correction": self.model_params.param.get("aggr_correction"),
        }
        self.model_samplerate = self.model_params.param["sr"]
        self.model_run = None

    def load_model(self):
        nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
        vr_5_1_models = [56817, 218409]
        model_size = math.ceil(os.stat(self.model_path).st_size / 1024)
        nn_arch_size = min(nn_arch_sizes, key=lambda size: abs(size - model_size))
        self.logger.debug(f"VR model size: {model_size}, architecture size: {nn_arch_size}")

        if nn_arch_size in vr_5_1_models or self.is_vr_51_model:
            self.model_run = nets_new.CascadedNet(
                self.model_params.param["bins"] * 2,
                nn_arch_size,
                nout=self.model_capacity[0],
                nout_lstm=self.model_capacity[1],
            )
            self.is_vr_51_model = True
        else:
            self.model_run = nets.determine_model_capacity(self.model_params.param["bins"] * 2, nn_arch_size)

        try:
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(self.model_path, map_location="cpu")
        except Exception:
            state_dict = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.model_run.load_state_dict(state_dict)
        self.model_run.eval()
        if self.fuse_conv_bn:
            fused = _fuse_sequential_conv_bn(self.model_run)
            self.logger.debug(f"Fused {fused} VR Conv2d+BatchNorm2d pairs")
        self.model_run.to(self.torch_device)
        if self.use_channels_last:
            self.model_run.to(memory_format=torch.channels_last)
        self.model_run.eval()

    def to(self, device):
        self.torch_device = device
        if self.model_run is not None:
            self.model_run.to(device)
            if self.use_channels_last:
                self.model_run.to(memory_format=torch.channels_last)
        return self

    def eval(self):
        if self.model_run is not None:
            self.model_run.eval()
        return self

    def process_stem(self, stem_source, spec):
        if not isinstance(stem_source, np.ndarray):
            stem_source = self.spec_to_wav(spec).T
            if self.model_samplerate != 44100:
                stem_source = spec_utils.resample_audio(stem_source.T, orig_sr=self.model_samplerate, target_sr=44100).T
        return stem_source.astype(np.float32, copy=False)

    def loading_mix(self, mix, sample_rate):
        x_wave, x_spec_s = {}, {}
        bands_n = len(self.model_params.param["band"])
        base_wave = self._ensure_stereo(mix)

        iterator = tqdm(range(bands_n, 0, -1), leave=False, desc="Processing VR bands") if self.debug else range(bands_n, 0, -1)
        for d in iterator:
            bp = self.model_params.param["band"][d]
            wav_resolution = "polyphase" if self.torch_device_mps is not None else bp["res_type"]
            if d == bands_n:
                x_wave[d] = self._resample_wave(base_wave, sample_rate, bp["sr"], wav_resolution)
                x_spec_s[d] = spec_utils.wave_to_spectrogram(
                    x_wave[d],
                    bp["hl"],
                    bp["n_fft"],
                    self.model_params,
                    band=d,
                    is_v51_model=self.is_vr_51_model,
                    torch_device=self.torch_device,
                )
            else:
                x_wave[d] = spec_utils.resample_audio(x_wave[d + 1], orig_sr=self.model_params.param["band"][d + 1]["sr"], target_sr=bp["sr"], res_type=wav_resolution)
                x_spec_s[d] = spec_utils.wave_to_spectrogram(
                    x_wave[d],
                    bp["hl"],
                    bp["n_fft"],
                    self.model_params,
                    band=d,
                    is_v51_model=self.is_vr_51_model,
                    torch_device=self.torch_device,
                )

            if d == bands_n and self.high_end_process:
                self.input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (self.model_params.param["pre_filter_stop"] - self.model_params.param["pre_filter_start"])
                self.input_high_end = x_spec_s[d][:, bp["n_fft"] // 2 - self.input_high_end_h:bp["n_fft"] // 2, :]

        return spec_utils.combine_spectrograms(x_spec_s, self.model_params, is_v51_model=self.is_vr_51_model)

    def _ensure_stereo(self, mix):
        mix = np.asarray(mix, dtype=np.float32)
        if mix.ndim == 1:
            return np.asfortranarray([mix, mix])
        if mix.shape[0] == 2:
            return np.asfortranarray(mix)
        if mix.shape[-1] == 2:
            return np.asfortranarray(mix.T)
        return np.asfortranarray([mono := np.mean(mix, axis=0), mono])

    @staticmethod
    def _resample_wave(wave, orig_sr, target_sr, res_type):
        return np.asfortranarray(wave) if int(orig_sr) == int(target_sr) else spec_utils.resample_audio(wave, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)

    def spec_to_wav(self, spec):
        if self.high_end_process and isinstance(self.input_high_end, np.ndarray) and self.input_high_end_h:
            input_high_end = spec_utils.mirroring("mirroring", spec, self.input_high_end, self.model_params)
            return spec_utils.cmb_spectrogram_to_wave(
                spec,
                self.model_params,
                self.input_high_end_h,
                input_high_end,
                is_v51_model=self.is_vr_51_model,
                torch_device=self.torch_device,
            )
        return spec_utils.cmb_spectrogram_to_wave(spec, self.model_params, is_v51_model=self.is_vr_51_model, torch_device=self.torch_device)
