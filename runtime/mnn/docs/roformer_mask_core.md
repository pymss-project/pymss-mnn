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

For fewer mask-head sessions without using the high-memory full-core graph,
export grouped mask bands. `--mask-group-size 8` reduces BSR mask sessions from
62 to 8 and MBR mask sessions from 60 to 8 while preserving exact output against
the per-band export:

```sh
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc --out-dir benchmark_results/mnn_work --only mask_bands --mask-group-size 8 --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset mbr_deux --out-dir benchmark_results/mnn_work --only mask_bands --mask-group-size 8 --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc --out-dir benchmark_results/mnn_work --only manifest --mask-group-size 8 --time-batch 1 --freq-batch 16
python tools/mnn_export/export_expr_micro_segments.py --preset mbr_deux --out-dir benchmark_results/mnn_work --only manifest --mask-group-size 8 --time-batch 1 --freq-batch 16
```

Export a fixed-shape full mask core when maximum throughput is more important
than the lowest peak memory:

```sh
python tools/mnn_export/export_roformer_mask_core.py \
  --preset bsr_hyperace_voc \
  --out-dir benchmark_results/mnn_core_fused
```

Run the C++ separator with the full-core graph:

```sh
runtime/mnn/build/mss_mnn_roformer_separate \
  --preset bsr_hyperace_voc \
  --core-model benchmark_results/mnn_core_fused/bsr_hyperace_voc/bsr_hyperace_voc_mask_core.mnn \
  --metadata benchmark_results/mnn_core_fused/bsr_hyperace_voc/bsr_hyperace_voc_metadata.json \
  --input benchmark_results/mnn_work/cpp_e2e/input/test_3s.wav \
  --output-dir benchmark_results/mnn_work/cpp_e2e/bsr_core_metal \
  --backend metal \
  --precision high \
  --threads 1
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
`"attention_op": "mnn"` in `manifest.json`. When the transformer input batch is
larger than one, the exporter emits one batch=1 MNN `Attention` op per
independent sequence and concatenates the results, because MNN's native
`Attention` batch dimension is not numerically reliable for RoFormer. Linear,
gate, FFN, and residual work remains batched. `--transformer-split
attention_ffn` exports each transformer as `*_attn.mnn` and `*_ffn.mnn`, letting
the C++ runtime choose precision per op group. The C++ runtime reads the
manifest and sets
`Interpreter::ATTENTION_OPTION=8` for `layer_*_attn` sessions by default. Pass
`--attention-kernel simple|flash` to the C++ separator to choose
`ATTENTION_OPTION=0` or `8` for native-attention segments. The linked MNN SDK
must be built with `-DMNN_SUPPORT_TRANSFORMER_FUSE=ON`; otherwise C++
execution fails with unsupported `Attention`.

On Metal/Auto, split native-attention exports use the validated per-family
policy. BSR-family `*_attn` sessions can run in `Normal`/FP16 while `*_ffn`
stays `High`; MelBandRoFormer `*_attn` and `*_ffn` stay `High` because FP16
attention accumulates too much end-to-end error. Unsplit native-attention
transformer sessions are also forced to `High`. MNN's current Metal fused path
is not exposed here because it needs SDK source changes for correct non-causal
RoFormer attention; this repository keeps the open-source runtime on unmodified
MNN SDKs.

The batch-unrolled native-attention exporter also works inside block graphs. A
BSR `frames=180` `block_00` probe with shape `[1, 180, 62, 256]` matched the
PyTorch block at high precision with `rmse=1.83e-4` (`ref_rms=0.731`, including
the known GELU approximation difference). The same block graph runs with
unmodified MNN Metal flash attention.

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
`--segment-cache mask-heads` on block exports to reuse the smaller
band/mask/segm sessions without keeping large transformer blocks resident,
`--segment-cache all` for maximum throughput, or `--segment-cache none` only
when profiling backend allocation behavior.

The Expr exporter maps RoFormer FFN GELU to MNN's native `UnaryOp GELU`.
This avoids the Metal fallback triggered by the previous `ERF`-based exact
formula. MNN's `GELU` is the tanh approximation, so keep block or end-to-end
quality checks in the validation loop when refreshing exported artifacts.

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

The same machine on a 60-second `test.m4a` slice produced the following for
split native-attention exports:

```text
bsr_hyperace_voc, metal-fast:
  time=201.76s, RTF=3.36, speed=0.30x realtime
  peak_footprint=2.51GB, max_rss=0.97GB

bsr_hyperace_voc, metal-autocast:
  time=203.81s, RTF=3.40, speed=0.29x realtime
  peak_footprint=3.09GB, max_rss=1.29GB

mbr_deux, metal-fast:
  time=249.49s, RTF=4.16, speed=0.24x realtime
  peak_footprint=4.61GB, max_rss=1.48GB

mbr_deux, metal-autocast:
  time=242.70s, RTF=4.05, speed=0.25x realtime
  peak_footprint=7.38GB, max_rss=3.11GB
```

For throughput, session count currently matters more than native `Attention`.
The manual/fused export keeps `freq_batch=16`, so it runs far fewer MNN sessions
than split native attention with `freq_batch=1`:

```text
bsr_hyperace_voc manual/fused, metal-fast:
  time=132.85s, RTF=2.21, speed=0.45x realtime
  peak_footprint=2.45GB, max_rss=1.10GB

mbr_deux manual/fused, metal-fast:
  time=190.52s, RTF=3.18, speed=0.31x realtime
  peak_footprint=4.70GB, max_rss=1.75GB

bsr_hyperace_voc manual/fused + mask_group_size=8, metal-fast:
  time=126.83s, RTF=2.11, speed=0.47x realtime
  peak_footprint=2.44GB, max_rss=1.49GB
  vs per-band output: max_abs=0, rmse=0

mbr_deux manual/fused + mask_group_size=8, metal-fast:
  time=158.55s, RTF=2.64, speed=0.38x realtime
  peak_footprint=4.40GB, max_rss=2.53GB
  vs per-band output: max_abs=0, rmse=0

mbr_deux manual/fused + mask_group_size=8 + cached mask groups, metal-fast:
  time=135.89s, RTF=2.26, speed=0.44x realtime
  peak_footprint=4.35GB, max_rss=2.16GB
  vs uncached grouped output: max_abs=0, rmse=0

bsr_hyperace_voc manual/fused + mask_group_size=8 + cached mask groups, metal-fast:
  time=119.49s, RTF=1.99, speed=0.50x realtime
  peak_footprint=2.08GB, max_rss=1.12GB
  vs uncached grouped output: max_abs=0, rmse=0
```

Profiling shows the remaining bottleneck is transformer sessions, not DSP:
MBR grouped/cached spends about `76.7s` in time transformer sessions and `50.0s`
in frequency transformer sessions out of `134.7s` total. BSR grouped/cached
spends about `61.8s` in time transformer sessions and `51.6s` in frequency
transformer sessions out of `118.8s` total. STFT, ISTFT, overlap-add, and
mask application together are below 1.5 seconds per 60-second file.

Use the C++ CLI `--profile` flag to print the same per-stage timing table.

`time_batch > 1` is implemented in the C++ runner but is not a validated Metal
speed path yet. On the same 60-second BSR slice, `time_batch=2` finished in
`107.27s` but produced all-NaN audio, and `time_batch=4` exited abnormally after
`111.41s`. Keep `time_batch=1` for release-quality Metal exports until a target
backend has separate correctness validation.

Block-core exports are the next reduction step for the small-session bottleneck:

```bash
python tools/mnn_export/export_expr_micro_segments.py \
  --preset bsr_hyperace_voc \
  --out-dir benchmark_results/mnn_work \
  --transformer-block-size 1 \
  --transformer-block-mode batched \
  --mask-group-size 8 \
  --time-batch 1 \
  --freq-batch 16
```

The batched block path expands rotary constants along the batch axis explicitly;
this avoids the MNN broadcast error that made earlier `time_batch > 1` and
naive block exports numerically invalid. A one-block CPU check for
`bsr_hyperace_voc` matched PyTorch at `rmse=7.34e-07`. The diagnostic
`--transformer-block-mode unrolled` keeps the original band-wise time
transformer semantics inside one graph, but it creates very large model files
and should not be used as the mobile default. The C++ runtime does not cache
`block_*.mnn` sessions by default to avoid 16GB-class memory pressure.

For mobile memory work, reduce the fixed frame count instead of only changing
session granularity:

```bash
python tools/mnn_export/export_expr_micro_segments.py \
  --preset bsr_hyperace_voc \
  --variant-name bsr_hyperace_voc_f180 \
  --frames 180 \
  --out-dir benchmark_results/mnn_work \
  --transformer-block-size 1 \
  --transformer-block-mode batched \
  --mask-group-size 8
```

For `bsr_hyperace_voc`, `--frames 180` produces metadata shape
`[1, 2050, 180, 2]`, `chunk_size=91648`, and `overlap_size=4582`. A 1-second
Metal smoke test with `precision=Normal`, `segment_cache=None`, and block
segments finished with `peak memory footprint=1.93GB`, below the 2GB mobile
target. The steady-state RTF estimate is about `2.24` (`~0.45x` realtime) from
the 1-second smoke profile. Time attention memory scales with `frames^2`, so
this cuts the main attention working set to roughly 3.7% of the default
938-frame export per chunk. Larger probes did not meet the same local target:
`frames=256` exited abnormally at `peak memory footprint=3.17GB`, `frames=192`
exited abnormally at `2.10GB`, and `frames=184` exited abnormally at `1.98GB`
on the 16GB M4 MacBook Air test machine.

The historical unsplit native-attention `freq_batch=16` probe was rejected:
it ran `bsr_hyperace_voc` in `143.87s` but changed the 60-second vocals output
by `rmse=1.36e-01`, so current native `Attention` exports must keep
`freq_batch=1` for correctness.

The full-core BSR graph cuts each chunk to one MNN session. With the freshly
exported `benchmark_results/mnn_core_fused/bsr_hyperace_voc` graph:

```text
bsr_hyperace_voc full mask core, Metal High:
  time=64.57s, RTF=1.08, speed=0.93x realtime
  peak_footprint=5.92GB, max_rss=1.04GB
  vs manual/fused: mean_abs=3.07e-04, rmse=5.52e-04

bsr_hyperace_voc full mask core, Metal Normal:
  time=49.81s, RTF=0.83, speed=1.20x realtime
  peak_footprint=3.48GB, max_rss=1.12GB
  rejected: vs High rmse=2.00e-02
```

The MBR full-core graph currently converts but returns all-zero MNN output on a
random mask-core input, so it is not a valid speed path yet.
