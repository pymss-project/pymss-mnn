# RoFormer MNN Mask Core

This runtime intentionally keeps audio DSP outside the MNN graph:

- host code decodes audio, chunks/overlaps, computes STFT/ISTFT, and folds chunks
- MNN runs only the RoFormer mask core
- complex tensors cross the boundary as real/imag float32 tensors

Supported first-stage presets:

- `bsr_hyperace_voc`: `models/BS-Roformer-HyperACE_v2_voc`, input shape `1,2050,938,2`
- `bsr_hyperace_voc_full`: `models/BS-Roformer-HyperACE_v2_voc`, HyperACE `mask_mode=full`, input shape `1,2050,938,2`
- `bsr_hyperace_voc_segm_only`: `models/BS-Roformer-HyperACE_v2_voc`, HyperACE `mask_mode=segm_only`, input shape `1,2050,938,2`
- `mbr_deux`: `models/mel-band-roformer-deux`, input shape `1,3958,1301,2`
- `logic_bs_roformer`: `models/logic_bs_roformer`, input shape `1,2050,1723,2`
- `bs_roformer_ep_368`: `models/bs_roformer_ep_368`, input shape `1,2050,801,2`
- `mel_band_roformer_big`: `models/Mel-Band-Roformer-big`, input shape `1,3958,1536,2`

Remaining local non-RoFormer models are documented separately in
`runtime/mnn/docs/non_roformer_mnn_validation.md`. They use
`tools/mnn_export/export_non_roformer_core.py`,
`tools/mnn_export/validate_non_roformer_core.py`, and
`tools/mnn_export/separate_non_roformer_with_mnn.py`.

Downloaded VR models are documented in
`runtime/mnn/docs/vr_mnn_validation.md`. They use
`tools/mnn_export/export_vr_core.py`, `tools/mnn_export/validate_vr_core.py`,
and `tools/mnn_export/separate_vr_with_mnn.py`.

Export the low-memory micro segments used by the default Python runner:

```sh
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc_full --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc_segm_only --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset mbr_deux --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset logic_bs_roformer --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset bs_roformer_ep_368 --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset mel_band_roformer_big --out-dir benchmark_results/mnn_work --time-batch 1 --freq-batch 16
```

Refresh manifests after partial re-export:

```sh
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc --out-dir benchmark_results/mnn_work --only transformers --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset mbr_deux --out-dir benchmark_results/mnn_work --only transformers --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc --out-dir benchmark_results/mnn_work --only manifest --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset mbr_deux --out-dir benchmark_results/mnn_work --only manifest --time-batch 1 --freq-batch 16
```

Export native MNN `Attention` transformer segments for flash attention:

```sh
python tools/mnn_export/export_expr_micro_segments.py \
  --preset bsr_hyperace_voc \
  --out-dir benchmark_results/mnn_flash_attention \
  --time-batch 1 \
  --freq-batch 1 \
  --attention-op mnn \
  --transformer-split attention_ffn

python tools/mnn_export/export_expr_micro_segments.py \
  --preset mbr_deux \
  --out-dir benchmark_results/mnn_flash_attention \
  --time-batch 1 \
  --freq-batch 1 \
  --attention-op mnn \
  --transformer-split attention_ffn
```

`--attention-op mnn` replaces each transformer segment's
`MatMul -> Softmax -> MatMul` attention with one MNN `Attention` op and records
`"attention_op": "mnn"` in `manifest.json`. `--transformer-split
attention_ffn` exports each transformer as `*_attn.mnn` and `*_ffn.mnn`, letting
the C++ runtime choose precision per op group. Keep `time_batch=1` and
`freq_batch=1`: current MNN `Attention` gives incorrect results for the
RoFormer frequency segments when this export batches multiple independent
sequences. The C++ runtime reads the manifest and sets
`Interpreter::ATTENTION_OPTION=8` for `layer_*_attn` sessions. The linked MNN SDK
must be built with `-DMNN_SUPPORT_TRANSFORMER_FUSE=ON`; otherwise C++ execution
fails with unsupported `Attention`.

On Metal/Auto, split native-attention exports use the validated per-family
policy. BSR-family `*_attn` sessions can run in `Normal`/FP16 while `*_ffn`
stays `High`; MelBandRoFormer `*_attn` and `*_ffn` stay `High` because FP16
attention accumulates too much end-to-end error. Unsplit native-attention
transformer sessions are also forced to `High`. Do not use MNN's
`attention_option=16` fused Metal variant for these segments until it has
separate quality validation.

Validate one mask-core chunk:

```sh
python tools/mnn_export/validate_expr_micro_segments.py --preset bsr_hyperace_voc --out-dir benchmark_results/mnn_work --threads 1 --skip-torch
python tools/mnn_export/validate_expr_micro_segments.py --preset bsr_hyperace_voc_full --out-dir benchmark_results/mnn_work --threads 1 --skip-torch
python tools/mnn_export/validate_expr_micro_segments.py --preset bsr_hyperace_voc_segm_only --out-dir benchmark_results/mnn_work --threads 1 --skip-torch
python tools/mnn_export/validate_expr_micro_segments.py --preset mbr_deux --out-dir benchmark_results/mnn_work --threads 1 --skip-torch
python tools/mnn_export/validate_expr_micro_segments.py --preset logic_bs_roformer --out-dir benchmark_results/mnn_work --threads 1 --skip-torch
python tools/mnn_export/validate_expr_micro_segments.py --preset bs_roformer_ep_368 --out-dir benchmark_results/mnn_work --threads 1 --skip-torch
python tools/mnn_export/validate_expr_micro_segments.py --preset mel_band_roformer_big --out-dir benchmark_results/mnn_work --threads 1 --skip-torch
```

Run end-to-end validation through PyMNN micro segments:

```sh
python tools/mnn_export/separate_with_mnn.py --preset bsr_hyperace_voc --audio test.m4a --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset bsr_hyperace_voc_full --audio test.m4a --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset bsr_hyperace_voc_segm_only --audio test.m4a --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset mbr_deux --audio test.m4a --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset logic_bs_roformer --audio test.m4a --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset bs_roformer_ep_368 --audio test.m4a --compare-torch
python tools/mnn_export/separate_with_mnn.py --preset mel_band_roformer_big --audio test.m4a --compare-torch
```

For MelBandRoFormer presets, the mask-core input is the `freq_indices` selected
STFT tensor; the host runtime scatters the mask back to the full frequency axis
after MNN.

For HyperACE segm presets, `segm_*.mnn` receives the full transformer output as
`[1, frames, bands, dim]`. `mask_mode=full` adds the segm output to the regular
per-band mask estimator, while `mask_mode=segm_only` skips mask-band execution.

Build the C++ mask-core runner after installing an MNN SDK with headers:

```sh
cmake -S runtime/mnn -B runtime/mnn/build -DMNN_ROOT=/path/to/mnn/install
cmake --build runtime/mnn/build
```

The reusable C++ runtime API is documented in
`runtime/mnn/docs/cpp_runtime.md`. For mobile library builds, pass
`-DMSS_MNN_BUILD_APPS=OFF` and link the `mss_mnn_core` target from the
platform project.

Prepared tensor smoke for one exported micro segment:

```sh
runtime/mnn/build/mss_mnn_mask_core \
  --model benchmark_results/mnn_work/bsr_hyperace_voc/expr_micro_segments/mask_band_00.mnn \
  --input mask_band_00_input.f32 \
  --shape 1,938,256 \
  --output mask_band_00_output.f32
```

This machine has a local MNN 3.5.0 CPU SDK build at
`/Volumes/2T/cache/mnn_sdk/MNN-install`. Use it with:

```sh
cmake -S runtime/mnn -B runtime/mnn/build -DMNN_ROOT=/Volumes/2T/cache/mnn_sdk/MNN-install
cmake --build runtime/mnn/build
```

Torch-free C++ separation uses already-decoded float WAV input. Decode `test.m4a`
with ffmpeg, then run the RoFormer separator CLI:

```sh
ffmpeg -y -i test.m4a -t 3 -ar 44100 -ac 2 -c:a pcm_f32le benchmark_results/mnn_work/cpp_e2e/input/test_3s.wav

DYLD_LIBRARY_PATH=/Volumes/2T/cache/mnn_sdk/MNN-install/lib:$DYLD_LIBRARY_PATH \
runtime/mnn/build/mss_mnn_roformer_separate \
  --preset bsr_hyperace_voc \
  --segments benchmark_results/mnn_work/bsr_hyperace_voc/expr_micro_segments \
  --metadata benchmark_results/mnn_work/bsr_hyperace_voc/bsr_hyperace_voc_metadata.json \
  --input benchmark_results/mnn_work/cpp_e2e/input/test_3s.wav \
  --output-dir benchmark_results/mnn_work/cpp_e2e/bsr_short \
  --backend cpu \
  --threads 1
```

For mobile-first Metal validation, keep the default `metal-fast` policy. MNN's
public C++ runtime sets precision per session, so the micro-segment runtime
applies the policy at segment/op-group level. For manual attention exports, only
`band_split` runs in high precision while transformer `layer_*` and `mask_*`
segments stay normal precision to favor FP16 throughput and lower memory. For
split native MNN `Attention` exports, BSR-family `*_attn` sessions stay normal
precision while `*_ffn` stays high precision; MelBandRoFormer `*_attn` and
`*_ffn` stay high precision. Use `--precision normal --precision-policy uniform`
for the absolute lowest-memory manual-attention run, `--precision-policy
metal-autocast` for tighter CPU/Metal parity, and `--precision high
--precision-policy uniform` for a full high-precision reference run. Use
`--segment-cache transformers` to avoid keeping mask/segm sessions resident,
`--segment-cache all` for maximum throughput, or `--segment-cache none` only
when profiling backend allocation behavior.

```sh
runtime/mnn/build/mss_mnn_roformer_separate \
  --preset bsr_hyperace_voc \
  --segments benchmark_results/mnn_work/bsr_hyperace_voc/expr_micro_segments \
  --metadata benchmark_results/mnn_work/bsr_hyperace_voc/bsr_hyperace_voc_metadata.json \
  --input benchmark_results/mnn_work/cpp_e2e/input/test_3s.wav \
  --output-dir benchmark_results/mnn_work/cpp_e2e/bsr_short_metal \
  --backend metal \
  --precision auto \
  --precision-policy metal-fast \
  --segment-cache transformers \
  --threads 1
```

On the local M4 MacBook Air with the GPU-enabled transformer-fuse MNN SDK,
split native MNN `Attention` exports, and `test_3s.wav`, CPU vs Metal produced:

```text
bsr_hyperace_voc, metal-fast, Attention FP16 + FFN High + mask FP16:
  time=29.14s, peak_footprint=1.55GB
  vocals vs CPU: max_abs=1.28e-06, mean_abs=1.24e-07, rmse=1.71e-07

bsr_hyperace_voc, metal-autocast, Attention FP16 + FFN High + mask High:
  time=28.82s, peak_footprint=1.65GB
  vocals vs CPU: max_abs=2.16e-07, mean_abs=2.12e-08, rmse=2.93e-08

bsr_hyperace_voc, full High reference:
  time=29.45s, peak_footprint=1.74GB
  vocals vs CPU: max_abs=1.39e-09, mean_abs=1.49e-10, rmse=2.07e-10

mbr_deux, metal-fast, Attention High + FFN High + mask FP16:
  time=35.84s, peak_footprint=3.15GB
  Vocals vs CPU:       max_abs=1.39e-06, mean_abs=9.69e-08, rmse=1.35e-07
  Instrumental vs CPU: max_abs=1.85e-03, mean_abs=1.81e-04, rmse=2.47e-04

mbr_deux, metal-autocast/full High, Attention High + FFN High + mask High:
  time=34.97s, peak_footprint=3.72GB
  Vocals vs CPU:       max_abs=1.08e-09, mean_abs=1.16e-10, rmse=1.62e-10
  Instrumental vs CPU: max_abs=2.76e-07, mean_abs=2.74e-08, rmse=3.98e-08
```

More aggressive FP16 settings were rejected:

```text
band_split Normal/FP16 with split native Attention:
  bsr_hyperace_voc vocals rmse=5.50e-02
  mbr_deux Vocals rmse=5.54e-02, Instrumental rmse=9.52e-02

layer_00 FFN Normal/FP16, single-segment CPU High vs Metal Normal:
  bsr time_ffn rmse=1.06e-03, freq_ffn rmse=1.24e-03
  mbr time_ffn rmse=4.43e-03, freq_ffn rmse=4.14e-03
```
