# VR MNN Validation Report

Date: 2026-05-28

This report records macOS validation for two downloaded VR models. Audio
resampling, STFT/ISTFT, band combination, padding, aggressiveness adjustment,
and stem reconstruction stay outside MNN. MNN runs the fixed-shape
`predict_mask` neural core.

Downloaded weights are stored under ignored `models/downloaded_vr/`.

## Models

| Preset | Source model | Architecture | Input tensor | Output tensor |
| --- | --- | --- | --- | --- |
| `vr_denoise_lite` | `UVR-DeNoise-Lite.pth` | VR 5.1 `CascadedNet`, LSTM path | `mag_patch [1,2,1025,512]` | `mask_patch [1,2,1025,384]` |
| `vr_mgm_main` | `MGM_MAIN_v4.pth` | legacy `CascadedASPPNet` | `mag_patch [1,2,1025,512]` | `mask_patch [1,2,1025,256]` |

Both presets use `batch_size=1`, `window_size=512`, no TTA, no post-process,
and CPU `threads=1` for validation.

## Artifacts

Artifacts were generated under ignored `benchmark_results/mnn_work/`.

| Preset | ONNX | MNN | Metadata |
| --- | ---: | ---: | --- |
| `vr_denoise_lite` | 18 MB | 19 MB | `benchmark_results/mnn_work/vr_denoise_lite/vr_denoise_lite_metadata.json` |
| `vr_mgm_main` | 32 MB | 32 MB | `benchmark_results/mnn_work/vr_mgm_main/vr_mgm_main_metadata.json` |

## Core Validation

Random fixed-shape validation compares PyTorch, ONNX Runtime, and MNN on CPU.

| Preset | MNN vs PyTorch mean_abs | MNN vs PyTorch rmse | ref_rms |
| --- | ---: | ---: | ---: |
| `vr_denoise_lite` | `1.5960e-09` | `3.7800e-08` | `4.5196e-04` |
| `vr_mgm_main` | `2.0336e-05` | `6.1449e-05` | `9.1044e-01` |

## Audio Validation

Three-second `test.m4a` validation compares PyMNN output with the original
PyTorch `VRSeparator` path.

| Preset | Stem | mean_abs | rmse | MNN seconds | PyTorch seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| `vr_denoise_lite` | `Noise` | `2.2242e-09` | `3.7487e-09` | `0.60` | `0.65` |
| `vr_denoise_lite` | `No Noise` | `4.2332e-09` | `6.9771e-09` | `0.60` | `0.65` |
| `vr_mgm_main` | `Instrumental` | `1.8124e-04` | `3.1876e-04` | `1.50` | `2.82` |
| `vr_mgm_main` | `Vocals` | `1.8124e-04` | `3.1876e-04` | `1.50` | `2.82` |

The helper used by `separate_vr_with_mnn.py` was also checked against
`VRSeparator.separate_array` with the same Torch model. The Torch helper and
original path match at numerical noise level, so the MNN-vs-Torch rows above
measure MNN core/runtime differences rather than a rewritten host pipeline.

Output WAVs are under:

- `benchmark_results/mnn_work/vr_denoise_lite/audio_short`
- `benchmark_results/mnn_work/vr_mgm_main/audio_short`

## Commands

```sh
python - <<'PY'
from pymss.model_download import download_model
model_dir = "/Volumes/2T/pymss-mnn/models/downloaded_vr"
for name in ["UVR-DeNoise-Lite.pth", "MGM_MAIN_v4.pth"]:
    download_model(name, model_dir=model_dir, source="modelscope", timeout=60)
PY

python tools/mnn_export/export_vr_core.py --preset vr_denoise_lite --out-dir benchmark_results/mnn_work --convert-timeout 1800
python tools/mnn_export/export_vr_core.py --preset vr_mgm_main --out-dir benchmark_results/mnn_work --convert-timeout 1800

python tools/mnn_export/validate_vr_core.py --preset vr_denoise_lite --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_vr_core.py --preset vr_mgm_main --out-dir benchmark_results/mnn_work --threads 1

python tools/mnn_export/separate_vr_with_mnn.py --preset vr_denoise_lite --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/vr_denoise_lite/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_vr_with_mnn.py --preset vr_mgm_main --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/vr_mgm_main/audio_short --seconds 3 --threads 1 --compare-torch
```

## Runtime Status

These validations use Python/PyMNN runners. The MNN graph itself is the VR
neural `predict_mask` core. A torch-free C++ host implementation for VR DSP,
resampling, padding, aggressiveness adjustment, and reconstruction is still
future work.
