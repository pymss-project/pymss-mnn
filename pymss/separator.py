import gc
import os
import logging
import re
from contextlib import contextmanager, nullcontext
import torch
import numpy as np
import platform
from time import time

from .utils import get_model_from_config
from .logger import get_separation_logger, set_log_level
from .config import AttrDict


INFERENCE_PARAM_TARGETS = {
    'batch_size': 'inference',
    'overlap_size': 'inference',
    'chunk_size': 'audio',
    'normalize': 'inference',
    'mask_mode': 'inference',
    'window_size': 'inference',
    'aggression': 'inference',
    'enable_tta': 'inference',
    'enable_post_process': 'inference',
    'post_process_threshold': 'inference',
    'high_end_process': 'inference',
    'use_amp': 'inference',
    'cuda_attention_backend': 'inference',
    'fuse_conv_bn': 'inference',
    'use_channels_last': 'inference',
    'shifts': 'inference',
    'split': 'inference',
    'overlap': 'inference',
    'stem_batch_size': 'inference',
}
PASSTHROUGH_INFERENCE_PARAMS = frozenset({
    'normalize',
    'mask_mode',
    'enable_tta',
    'enable_post_process',
    'high_end_process',
    'use_amp',
    'cuda_attention_backend',
    'fuse_conv_bn',
    'use_channels_last',
    'split',
})
FAST_INIT_MODEL_TYPES = {'bs_roformer', 'bs_roformer_hyperace', 'mel_band_roformer'}
LEGACY_DEMUCS_MODEL_TYPES = {'demucs', 'tasnet', 'legacy_demucs', 'legacy_tasnet'}


def _resolve_public_device(device, inference_params, logger):
    inference_params = dict(inference_params or {})
    requested_device = device
    if requested_device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")
    return requested_device, inference_params


def _select_device(device, device_ids, logger):
    if device not in ['cpu', 'cuda', 'mps']:
        if torch.cuda.is_available():
            logger.debug("CUDA is available in Torch, setting Torch device to CUDA")
            return f'cuda:{device_ids[0]}'
        if torch.backends.mps.is_available():
            logger.debug("Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS")
            return "mps"
        return "cpu"

    if device == "cpu":
        logger.warning("No hardware acceleration could be configured, running in CPU mode")
    return device


def _unwrap_state_dict(state_dict):
    for key in ('state', 'state_dict', 'model_state_dict'):
        if key in state_dict:
            return state_dict[key]
    return state_dict


def _apollo_state_dict_path(model_path):
    root, ext = os.path.splitext(model_path)
    candidates = []
    if ext:
        candidates.append(f"{root}.pymss_state_dict.pt")
    candidates.append(f"{model_path}.pymss_state_dict.pt")
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return model_path


def _load_state_dict(model_type, model_path, device):
    if model_type == 'vr':
        return None
    map_location = "cpu"
    if model_type == 'htdemucs':
        stubbed_modules = _install_demucs_pickle_stubs()
        try:
            state_dict = torch.load(model_path, map_location=map_location, weights_only=False)
        finally:
            _restore_modules(stubbed_modules)
        return _unwrap_state_dict(state_dict)
    if model_type == 'apollo':
        model_path = _apollo_state_dict_path(model_path)
        return _unwrap_state_dict(torch.load(model_path, map_location=map_location, weights_only=False))
    try:
        return _unwrap_state_dict(torch.load(model_path, map_location=map_location, weights_only=True, mmap=True))
    except (TypeError, ValueError, RuntimeError):
        return _unwrap_state_dict(torch.load(model_path, map_location=map_location, weights_only=True))


@contextmanager
def _skip_torch_default_init():
    classes = (
        torch.nn.Linear,
        torch.nn.Bilinear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.Embedding,
        torch.nn.EmbeddingBag,
        torch.nn.RNN,
        torch.nn.GRU,
        torch.nn.LSTM,
        torch.nn.MultiheadAttention,
    )
    saved = {cls: cls.reset_parameters for cls in classes if hasattr(cls, 'reset_parameters')}
    try:
        for cls in saved:
            cls.reset_parameters = lambda self: None
        yield
    finally:
        for cls, reset_parameters in saved.items():
            cls.reset_parameters = reset_parameters


def _install_demucs_pickle_stubs():
    import sys
    import types

    module_names = ('demucs', 'demucs.demucs', 'demucs.hdemucs', 'demucs.htdemucs')
    previous = {name: sys.modules.get(name) for name in module_names}
    package = sys.modules.setdefault('demucs', types.ModuleType('demucs'))
    package.__path__ = []
    for module_name, class_names in {
        'demucs': ('Demucs',),
        'hdemucs': ('HDemucs', 'HTDemucs'),
        'htdemucs': ('HTDemucs',),
    }.items():
        full_name = f'demucs.{module_name}'
        module = sys.modules.setdefault(full_name, types.ModuleType(full_name))
        setattr(package, module_name, module)
        for class_name in class_names:
            if not hasattr(module, class_name):
                setattr(module, class_name, type(class_name, (), {'__module__': full_name}))
    return previous


def _restore_modules(previous):
    import sys

    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _runtime_model_type(model_type, state_dict):
    return 'bs_roformer_hyperace' if model_type == 'bs_roformer' and any('.segm.' in key for key in state_dict) else model_type


def _infer_mel_band_roformer_mlp_hidden_layers(state_dict):
    pattern = re.compile(r'(?:^|\.)mask_estimators\.0\.to_freqs\.0\.0\.(\d+)\.weight$')
    layer_indices = sorted({int(match.group(1)) for key in state_dict for match in [pattern.search(key)] if match})
    if not layer_indices:
        return None
    return len(layer_indices) - 1


def _coerce_mps_float64(module):
    for child in module.modules():
        for name, param in list(child._parameters.items()):
            if param is not None and param.dtype == torch.float64:
                child._parameters[name] = torch.nn.Parameter(param.detach().float(), requires_grad=param.requires_grad)
        for name, buffer in list(child._buffers.items()):
            if buffer is not None and buffer.dtype == torch.float64:
                child._buffers[name] = buffer.float()


def _model_is_stereo(model_type, config):
    if model_type == 'vr':
        return True
    if model_type in ['bs_roformer', 'bs_roformer_hyperace', 'mel_band_roformer', *LEGACY_DEMUCS_MODEL_TYPES]:
        return config.model.get("stereo", True)
    return True


def _prepare_mix_channels(mix, is_stereo, logger):
    if is_stereo and len(mix.shape) == 1:
        logger.warning("Track is mono, but model is stereo, adding a second channel.")
        return np.stack([mix, mix], axis=0)
    if is_stereo and len(mix.shape) > 2:
        logger.warning("Track has more than 2 channels, taking mean of all channels and adding a second channel.")
        mono = np.mean(mix, axis=0)
        return np.stack([mono, mono], axis=0)
    if not is_stereo and len(mix.shape) != 1:
        logger.warning("Track has more than 1 channels, but model is mono, taking mean of all channels.")
        return np.mean(mix, axis=0)
    return mix


class MSSeparator:
    def __init__(
            self,
            model_type,
            model_path,
            config_path = None,
            device = 'auto',
            device_ids = [0],
            output_format = 'wav',
            use_tta = False,
            store_dirs = 'results', # str for single folder, dict with instrument keys for multiple folders
            audio_params = {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k", "m4a_bit_rate": "192k", "m4a_aac_at_quality": 2},
            logger = None,
            debug = False,
            inference_params = {
                "batch_size": None,
                "overlap_size": None,
                "chunk_size": None,
                "normalize": None,
                "mask_mode": None,
            }
    ):

        if not model_type:
            raise ValueError('model_type is required')
        if not model_path:
            raise ValueError('model_path is required')

        logger = logger if logger is not None else get_separation_logger()
        device, inference_params = _resolve_public_device(device, inference_params, logger)

        self.model_type = model_type

        self.model_path = model_path
        self.config_path_given = config_path is not None
        self.config_path = config_path if config_path else (model_path + '.yaml')
        self.output_format = output_format
        self.use_tta = use_tta
        self.store_dirs = store_dirs
        self.audio_params = audio_params
        self.logger = logger
        self.debug = debug
        self.inference_params = inference_params

        if self.debug:
            set_log_level(self.logger, logging.DEBUG)
        else:
            set_log_level(self.logger, logging.INFO)

        self.log_system_info()

        self.device_ids = device_ids
        self.device = _select_device(device, self.device_ids, self.logger)

        torch.backends.cudnn.benchmark = True
        self.logger.info(f'Using device: {self.device}, device_ids: {self.device_ids}')

        self.model, self.config = self.load_model()

    def log_system_info(self):
        os_name = platform.system()
        os_version = platform.version()
        self.logger.debug(f"Operating System: {os_name} {os_version}")

        python_version = platform.python_version()
        self.logger.debug(f"Python Version: {python_version}")

        pytorch_version = torch.__version__
        self.logger.debug(f"PyTorch Version: {pytorch_version}")

    def load_model(self):
        start_time = time()
        if self.model_type == 'vr':
            from .modules.vocal_remover.vr_models import get_vr_model_metadata
            from .modules.vocal_remover import VRSeparator

            model_data = get_vr_model_metadata(self.model_path)
            instruments = [model_data["primary_stem"], model_data["secondary_stem"]]
            config = AttrDict({
                "training": {
                    "instruments": instruments,
                    "target_instrument": None,
                    "use_amp": True,
                },
                "audio": {
                    "sample_rate": 44100,
                },
                "inference": {
                    "batch_size": 2,
                    "window_size": 512,
                    "aggression": 5,
                    "enable_tta": self.use_tta,
                    "enable_post_process": False,
                    "post_process_threshold": 0.2,
                    "high_end_process": False,
                    "use_amp": True,
                    "fuse_conv_bn": False,
                    "use_channels_last": False,
                    "normalize": False,
                },
            })
            self.update_inference_params(config, self.inference_params)
            common_config = {
                "logger": self.logger,
                "debug": self.debug,
                "torch_device": self.device,
                "torch_device_cpu": torch.device("cpu"),
                "torch_device_mps": torch.device("mps") if torch.device(self.device).type == "mps" else None,
                "model_name": os.path.basename(self.model_path),
                "model_path": self.model_path,
                "model_data": model_data,
                "sample_rate": 44100,
                "callback": None,
            }
            model = VRSeparator(common_config, config.inference)
            model.load_model()
            self.logger.info(f"Model loader params: model_type: vr, model_path: {self.model_path}")
            self.logger.info(f"Model params: instruments: {config.training.instruments}, target_instrument: None")
            self.logger.debug(f"Loading VR model completed, duration: {time() - start_time:.2f} seconds")
            return model, config

        if self.model_type in LEGACY_DEMUCS_MODEL_TYPES:
            from .modules.legacy_demucs import load_legacy_demucs_model

            config_path = self.config_path if self.config_path_given else None
            model, config = load_legacy_demucs_model(self.model_path, config_path)
            config = AttrDict(config)
            self.update_inference_params(config, self.inference_params)
            model = model.to(self.device)
            model.eval()

            self.logger.info(f"Model loader params: model_type: {self.model_type}, model_path: {self.model_path}, config_path: {config_path}")
            self.logger.info(f"Model params: instruments: {config.training.get('instruments', None)}, target_instrument: {config.training.get('target_instrument', None)}")
            self.logger.debug(f"Model params: batch_size: {config.inference.get('batch_size', None)}, overlap_size: {config.inference.get('overlap_size', None)}, chunk_size: {config.audio.get('chunk_size', None)}, normalize: {config.inference.get('normalize', None)}, use_tta: {self.use_tta}")
            self.logger.debug(f"Loading legacy Demucs/TasNet model completed, duration: {time() - start_time:.2f} seconds")
            return model, config

        state_dict = _load_state_dict(self.model_type, self.model_path, self.device)
        model_type = _runtime_model_type(self.model_type, state_dict)
        model_kwargs_override = None
        if model_type == 'mel_band_roformer':
            model_kwargs_override = {
                'mlp_hidden_layers': _infer_mel_band_roformer_mlp_hidden_layers(state_dict),
            }

        init_context = _skip_torch_default_init() if model_type in FAST_INIT_MODEL_TYPES else nullcontext()
        with init_context:
            model, config = get_model_from_config(model_type, self.config_path, model_kwargs_override=model_kwargs_override)

        self.update_inference_params(config, self.inference_params)
        self.apply_model_inference_config(model, config)

        self.logger.info(f"Model loader params: model_type: {model_type}, model_path: {self.model_path}, config_path: {self.config_path}")
        self.logger.info(f"Model params: instruments: {config.training.get('instruments', None)}, target_instrument: {config.training.get('target_instrument', None)}")
        self.logger.debug(f"Model params: batch_size: {config.inference.get('batch_size', None)}, overlap_size: {config.inference.get('overlap_size', None)}, chunk_size: {config.audio.get('chunk_size', None)}, normalize: {config.inference.get('normalize', None)}, use_tta: {self.use_tta}")

        try:
            model.load_state_dict(state_dict, assign=True)
        except TypeError:
            model.load_state_dict(state_dict)
        if torch.device(self.device).type == "mps":
            _coerce_mps_float64(model)

        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        model = model.to(self.device)
        model.eval()

        self.logger.debug(f"Loading model completed, duration: {time() - start_time:.2f} seconds")
        return model, config

    def apply_model_inference_config(self, model, config):
        if hasattr(model, 'set_mask_mode'):
            model.set_mask_mode(config.inference.get('mask_mode', 'no_segm'))
        cuda_attention_backend = config.inference.get('cuda_attention_backend', None)
        if cuda_attention_backend is not None:
            for module in model.modules():
                if hasattr(module, 'set_cuda_attention_backend'):
                    module.set_cuda_attention_backend(cuda_attention_backend)

    def update_inference_params(self, config, params):
        for key, section in INFERENCE_PARAM_TARGETS.items():
            value = params.get(key)
            if value is None:
                continue
            if key not in PASSTHROUGH_INFERENCE_PARAMS:
                value = float(value) if key in {'post_process_threshold', 'overlap'} else int(value)
            config[section][key] = value
        return config

    def del_cache(self):
        self.logger.debug("Running garbage collection...")
        gc.collect()
        if "mps" in self.device:
            self.logger.debug("Clearing MPS cache...")
            torch.mps.empty_cache()
        if "cuda" in self.device:
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()
