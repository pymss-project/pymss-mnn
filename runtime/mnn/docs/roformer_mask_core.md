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
