# RoFormer MNN Validation Report

Date: 2026-05-28

This report records the first-stage macOS validation for running BSR and MBR
RoFormer mask cores through MNN. Audio DSP stays outside MNN: host code decodes
audio, prepares STFT chunks, runs overlap-add, and reconstructs audio. MNN runs
only the neural mask core.

## Models

| Preset | Source model | Model type | Input tensor | Output tensor |
| --- | --- | --- | --- | --- |
| `bsr_hyperace_voc` | `models/BS-Roformer-HyperACE_v2_voc` | `BSRoformerHyperACE` | `[1, 2050, 938, 2]` | `[1, 1, 2050, 938, 2]` |
| `bsr_hyperace_voc_full` | `models/BS-Roformer-HyperACE_v2_voc` | `BSRoformerHyperACE`, `mask_mode=full` | `[1, 2050, 938, 2]` | `[1, 1, 2050, 938, 2]` |
| `bsr_hyperace_voc_segm_only` | `models/BS-Roformer-HyperACE_v2_voc` | `BSRoformerHyperACE`, `mask_mode=segm_only` | `[1, 2050, 938, 2]` | `[1, 1, 2050, 938, 2]` |
| `mbr_deux` | `models/mel-band-roformer-deux` | `MelBandRoformer` | `[1, 3958, 1301, 2]` | `[1, 2, 3958, 1301, 2]` |
| `logic_bs_roformer` | `models/logic_bs_roformer` | `BSRoformer` | `[1, 2050, 1723, 2]` | `[1, 6, 2050, 1723, 2]` |
| `bs_roformer_ep_368` | `models/bs_roformer_ep_368` | `BSRoformer` | `[1, 2050, 801, 2]` | `[1, 1, 2050, 801, 2]` |
| `mel_band_roformer_big` | `models/Mel-Band-Roformer-big` | `MelBandRoformer` | `[1, 3958, 1536, 2]` | `[1, 1, 3958, 1536, 2]` |

For MelBandRoFormer presets, the host runner gathers `freq_indices` before MNN
and scatters the returned mask back onto the full frequency axis.

## Export Artifacts

Artifacts were generated under ignored `benchmark_results/mnn_work/`.

| Preset | Segment dir | Transformer segments | Mask-head segments | Total MNN files |
| --- | --- | ---: | ---: | ---: |
| `bsr_hyperace_voc` | `benchmark_results/mnn_work/bsr_hyperace_voc/expr_micro_segments` | 24 | 63 | 88 |
| `bsr_hyperace_voc_full` | `benchmark_results/mnn_work/bsr_hyperace_voc_full/expr_micro_segments` | 24 | 62 + 1 segm | 88 |
| `bsr_hyperace_voc_segm_only` | `benchmark_results/mnn_work/bsr_hyperace_voc_segm_only/expr_micro_segments` | 24 | 1 segm | 26 |
| `mbr_deux` | `benchmark_results/mnn_work/mbr_deux/expr_micro_segments` | 24 | 61 | 86 |
| `logic_bs_roformer` | `benchmark_results/mnn_work/logic_bs_roformer/expr_micro_segments` | 24 | 62 | 87 |
| `bs_roformer_ep_368` | `benchmark_results/mnn_work/bs_roformer_ep_368/expr_micro_segments` | 24 | 62 | 87 |
| `mel_band_roformer_big` | `benchmark_results/mnn_work/mel_band_roformer_big/expr_micro_segments` | 16 | 60 | 77 |

Both presets use `time_batch=1` and `freq_batch=16`.

## Numeric Validation

Random mask-core validation compares MNN micro-segment output with the PyTorch
wrapper on CPU float32.

| Preset | max_abs | mean_abs | rmse | ref_rms |
| --- | ---: | ---: | ---: | ---: |
| `bsr_hyperace_voc` | `3.5405e-05` | `9.06797e-07` | `1.94002e-06` | `0.704027` |
| `bsr_hyperace_voc_full` | `2.37465e-04` | `9.41991e-07` | `2.14414e-06` | `0.703924` |
| `bsr_hyperace_voc_segm_only` | `2.37593e-04` | `5.65488e-08` | `9.08814e-07` | `0.000860908` |
| `mbr_deux` | `6.43134e-05` | `1.44984e-06` | `3.12721e-06` | `0.661847` |
| `logic_bs_roformer` | `2.98023e-05` | `2.15687e-07` | `8.98563e-07` | `0.0657125` |
| `bs_roformer_ep_368` | `3.72827e-05` | `8.45458e-07` | `2.10510e-06` | `0.486252` |
| `mel_band_roformer_big` | `5.55068e-06` | `7.19661e-10` | `3.30526e-08` | `0.00234908` |

Three-second `test.m4a` end-to-end validation compares PyMNN micro segments
with the original PyTorch separator.

| Preset | Stem | mean_abs | rmse | MNN seconds | PyTorch seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| `bsr_hyperace_voc` | `vocals` | `4.667e-10` | `6.671e-10` | `84.96` | `8.86` |
| `bsr_hyperace_voc_full` | `vocals` | `5.048e-09` | `8.805e-09` | `34.65` | `25.71` |
| `bsr_hyperace_voc_segm_only` | `vocals` | `5.079e-09` | `8.884e-09` | `34.67` | `22.38` |
| `mbr_deux` | `Vocals` | `1.563e-09` | `2.798e-09` | `272.32` | `31.91` |
| `mbr_deux` | `Instrumental` | `3.796e-08` | `5.448e-08` | `272.32` | `31.91` |
| `logic_bs_roformer` | `bass` | `5.749e-10` | `1.001e-09` | `66.85` | `1959.07` |
| `logic_bs_roformer` | `drums` | `4.181e-10` | `6.230e-10` | `66.85` | `1959.07` |
| `logic_bs_roformer` | `other` | `5.548e-06` | `9.647e-06` | `66.85` | `1959.07` |
| `logic_bs_roformer` | `vocals` | `2.917e-10` | `4.018e-10` | `66.85` | `1959.07` |
| `logic_bs_roformer` | `guitar` | `9.587e-10` | `1.519e-09` | `66.85` | `1959.07` |
| `logic_bs_roformer` | `piano` | `5.589e-06` | `9.723e-06` | `66.85` | `1959.07` |
| `bs_roformer_ep_368` | `Vocals` | `1.538e-08` | `7.105e-08` | `57.18` | `29.50` |
| `mel_band_roformer_big` | `vocals` | `9.761e-11` | `1.441e-10` | `51.41` | `70.68` |

Full-length `test.m4a` MNN-only validation produced stereo float WAV files with
sample rate `44100`, `13742080` frames, and duration `311.611791` seconds.

| Preset | Output files | MNN seconds |
| --- | --- | ---: |
| `bsr_hyperace_voc` | `benchmark_results/mnn_work/bsr_hyperace_voc/audio_full/test_vocals.wav` | `1061.29` |
| `mbr_deux` | `benchmark_results/mnn_work/mbr_deux/audio_full/test_Vocals.wav`, `benchmark_results/mnn_work/mbr_deux/audio_full/test_Instrumental.wav` | `872.12` |

## Commands

```sh
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc_full --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc_segm_only --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset mbr_deux --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset logic_bs_roformer --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset bs_roformer_ep_368 --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset mel_band_roformer_big --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16

python tools/mnn_export/validate_expr_micro_segments.py --preset bsr_hyperace_voc --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_expr_micro_segments.py --preset bsr_hyperace_voc_full --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_expr_micro_segments.py --preset bsr_hyperace_voc_segm_only --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_expr_micro_segments.py --preset mbr_deux --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_expr_micro_segments.py --preset logic_bs_roformer --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_expr_micro_segments.py --preset bs_roformer_ep_368 --out-dir benchmark_results/mnn_work --threads 1
python tools/mnn_export/validate_expr_micro_segments.py --preset mel_band_roformer_big --out-dir benchmark_results/mnn_work --threads 1

python tools/mnn_export/separate_with_mnn.py --preset bsr_hyperace_voc --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/bsr_hyperace_voc/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset bsr_hyperace_voc_full --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/bsr_hyperace_voc_full/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset bsr_hyperace_voc_segm_only --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/bsr_hyperace_voc_segm_only/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset mbr_deux --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/mbr_deux/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset logic_bs_roformer --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/logic_bs_roformer/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset bs_roformer_ep_368 --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/bs_roformer_ep_368/audio_short --seconds 3 --threads 1 --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset mel_band_roformer_big --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/mel_band_roformer_big/audio_short --seconds 3 --threads 1 --compare-torch

python tools/mnn_export/separate_with_mnn.py --preset bsr_hyperace_voc --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/bsr_hyperace_voc/audio_full --threads 1
python tools/mnn_export/separate_with_mnn.py --preset mbr_deux --audio test.m4a --out-dir benchmark_results/mnn_work --store-dir benchmark_results/mnn_work/mbr_deux/audio_full --threads 4
```

## C++ Runtime Build

The `MNN` PyPI package used by the Python validation does not provide C++
headers. A matching MNN 3.5.0 CPU SDK was built from
`https://github.com/alibaba/MNN.git` tag `3.5.0` at commit `c35f14f` and
installed to `/Volumes/2T/cache/mnn_sdk/MNN-install`.

Build command:

```sh
cmake -S runtime/mnn -B runtime/mnn/build -DMNN_ROOT=/Volumes/2T/cache/mnn_sdk/MNN-install -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/mnn/build --parallel 4
```

For iOS/Android library builds, disable desktop apps and link `mss_mnn_core`
from the platform project:

```sh
cmake -S runtime/mnn -B runtime/mnn/build-mobile \
  -DMNN_ROOT=/path/to/MNN-install \
  -DMSS_MNN_BUILD_APPS=OFF \
  -DCMAKE_BUILD_TYPE=Release
```

C++ CLI smoke used `mask_band_00.mnn`:

```sh
DYLD_LIBRARY_PATH=/Volumes/2T/cache/mnn_sdk/MNN-install/lib:$DYLD_LIBRARY_PATH \
runtime/mnn/build/mss_mnn_mask_core \
  --model benchmark_results/mnn_work/bsr_hyperace_voc/expr_micro_segments/mask_band_00.mnn \
  --input benchmark_results/mnn_work/cpp_smoke/bsr_mask_band_00_input.f32 \
  --shape 1,938,256 \
  --output benchmark_results/mnn_work/cpp_smoke/bsr_mask_band_00_output.f32 \
  --input-name input \
  --output-name output \
  --threads 1
```

Smoke result: wrote `7504` float32 values with output shape `[1, 1, 938, 8]`.

## Torch-free C++ E2E Smoke

The C++ runtime now has a torch-free RoFormer separation CLI:
`runtime/mnn/build/mss_mnn_roformer_separate`. It reads already-decoded stereo
float WAV, performs STFT/ISTFT, chunk overlap-add, MBR gather/scatter, and MNN
micro-segment execution in C++.

The CLI is now a thin wrapper around the reusable `mss_mnn::RoformerSeparator`
C++ API documented in `runtime/mnn/docs/cpp_runtime.md`.

Validation input was decoded from `test.m4a`:

```sh
ffmpeg -y -v error -i test.m4a -t 3 -ar 44100 -ac 2 -c:a pcm_f32le benchmark_results/mnn_work/cpp_e2e/input/test_3s.wav
```

Three-second C++ output compared with the PyMNN micro-segment output:

| Preset | Stem | max_abs | mean_abs | rmse | ref_rms |
| --- | --- | ---: | ---: | ---: | ---: |
| `bsr_hyperace_voc` | `vocals` | `2.050e-07` | `1.572e-08` | `2.248e-08` | `4.789e-06` |
| `bsr_hyperace_voc_full` | `vocals` | `1.579e-06` | `1.590e-07` | `2.220e-07` | `3.347e-06` |
| `bsr_hyperace_voc_segm_only` | `vocals` | `1.547e-06` | `1.554e-07` | `2.173e-07` | `6.933e-06` |
| `mbr_deux` | `Vocals` | `5.427e-07` | `2.409e-08` | `4.489e-08` | `9.149e-06` |
| `mbr_deux` | `Instrumental` | `8.002e-06` | `4.390e-07` | `6.810e-07` | `7.355e-02` |
| `logic_bs_roformer` | `bass` | `4.034e-07` | `1.987e-08` | `3.044e-08` | `5.462e-07` |
| `logic_bs_roformer` | `drums` | `2.220e-07` | `1.188e-08` | `1.747e-08` | `5.450e-07` |
| `logic_bs_roformer` | `other` | `2.541e-03` | `1.372e-04` | `2.293e-04` | `1.173e-02` |
| `logic_bs_roformer` | `vocals` | `1.018e-07` | `8.562e-09` | `1.199e-08` | `2.899e-07` |
| `logic_bs_roformer` | `guitar` | `4.341e-07` | `2.915e-08` | `4.262e-08` | `9.705e-07` |
| `logic_bs_roformer` | `piano` | `2.608e-03` | `1.394e-04` | `2.332e-04` | `7.190e-02` |
| `bs_roformer_ep_368` | `Vocals` | `6.291e-06` | `1.113e-07` | `4.487e-07` | `5.539e-06` |
| `mel_band_roformer_big` | `vocals` | `6.262e-08` | `6.212e-09` | `9.646e-09` | `9.537e-08` |

The C++ rows compare the torch-free C++ output with the PyMNN micro-segment
output using the same decoded three-second float WAV input. Mask-core MNN vs
PyTorch validation above remains the tighter model-core correctness check.
