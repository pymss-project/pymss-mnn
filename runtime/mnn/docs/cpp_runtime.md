# Standalone C++ Runtime

The runtime path is C++ only. It links MNN C++ Runtime and consumes exported
`.mnn` files plus metadata generated offline by `tools/mnn_export`. It does not
load `pymss`, Python, PyMNN, Torch, NumPy, librosa, or av at inference time.

## Layout

```text
runtime/mnn/
  include/mss_mnn/
    audio.hpp
    mnn_mask_core.hpp
    roformer_separator.hpp
  src/
    audio.cpp
    mnn_mask_core.cpp
    roformer_separator.cpp
  apps/mac_cli/
```

`mss_mnn_core` is the reusable library target. Desktop CLIs are optional and
can be disabled for iOS/Android library builds:

```sh
cmake -S runtime/mnn -B runtime/mnn/build-ios \
  -DMNN_ROOT=/path/to/MNN-install \
  -DMSS_MNN_BUILD_APPS=OFF
```

## Public API

Core tensor execution:

```cpp
mss_mnn::MNNModelOptions options;
options.backend = mss_mnn::MNNBackend::CPU;
options.precision = mss_mnn::MNNPrecision::Auto;
options.threads = 1;

mss_mnn::MNNModel model("core.mnn", options);
mss_mnn::MNNTensor input;
input.name = "input";
input.shape = {1, 938, 256};
input.data = input_values;

std::vector<mss_mnn::MNNTensor> output =
    model.run({input}, {"output"});
```

Single-input/single-output mask-core convenience:

```cpp
mss_mnn::MaskCoreOptions options;
options.input_name = "input";
options.output_name = "output";
options.backend = mss_mnn::MNNBackend::CPU;
options.precision = mss_mnn::MNNPrecision::Auto;
options.threads = 1;

mss_mnn::MNNMaskCore core("mask_band_00.mnn", options);
std::vector<float> output = core.run(input_values, {1, 938, 256});
```

RoFormer end-to-end separation:

```cpp
mss_mnn::RoformerSeparatorOptions options;
options.segment_dir = "expr_micro_segments";
options.metadata_path = "bsr_hyperace_voc_metadata.json";
options.backend = mss_mnn::MNNBackend::CPU;
options.precision = mss_mnn::MNNPrecision::Auto;
options.precision_policy = mss_mnn::RoformerPrecisionPolicy::MetalFast;
options.segment_cache_policy = mss_mnn::RoformerSegmentCachePolicy::TransformersOnly;
options.threads = 1;

mss_mnn::RoformerSeparator separator(options);
mss_mnn::AudioBuffer input = mss_mnn::read_wav("input.wav");
std::vector<mss_mnn::AudioBuffer> stems = separator.separate(input);
```

For a fixed-shape full mask-core graph, set `core_model_path` instead of
`segment_dir`. This runs one MNN session per chunk and avoids the micro-segment
CPU/GPU round trips:

```cpp
mss_mnn::RoformerSeparatorOptions options;
options.core_model_path = "bsr_hyperace_voc_mask_core.mnn";
options.metadata_path = "bsr_hyperace_voc_metadata.json";
options.backend = mss_mnn::MNNBackend::Metal;
options.precision = mss_mnn::MNNPrecision::High;
```

For mobile apps, feed `AudioBuffer` directly from the platform audio decoder and
write platform-native output. The WAV helpers are only a portable CLI/testing
convenience.

## Backends

`MNNBackend` supports `CPU`, `Auto`, `Metal`, `OpenCL`, and `Vulkan`.
`MNNPrecision` supports `Auto`, `Normal`, `High`, `Low`, and `LowBF16`.
RoFormer also exposes `RoformerPrecisionPolicy`: `Uniform` applies one
precision to every segment, `MetalFast` keeps the mobile-first minimum high
precision set, and `MetalAutocast` raises additional mask segments for tighter
CPU/Metal parity.
`RoformerSegmentCachePolicy` controls how many micro-segment sessions stay
resident: `All` is fastest, `BlocksOnly` caches `block_*.mnn` plus
`band_split`, `TransformersOnly` avoids caching block/segm heads, and `None`
disables the runtime cache for diagnostics. On Metal, `None` can be slower and
may not lower peak footprint because the backend can retain internal
allocations.
On Apple platforms, the runtime wraps chunk and MNN segment execution in scoped
Objective-C autorelease pools. This drains temporary Metal objects during long
audio runs instead of letting them accumulate until process exit.
RoFormer manifests with `"mask_group_size" > 1` use grouped mask-band segments
(`mask_group_*.mnn`) to reduce mask-head session count while keeping the
transformer and DSP pipeline unchanged.
RoFormer manifests with `"transformer_block_size" > 0` use `block_*.mnn`
segments for whole RoFormer layer blocks. This removes the host-side loop over
thousands of small time/frequency transformer sessions. Block segments are kept
resident by `segment_cache_policy=BlocksOnly` or `All`; the default
`TransformersOnly` still loads one block at a time because a full set of
fixed-shape block sessions can exceed low-memory targets on larger frame
exports.
`TransformersOnly` also caches `mask_group_*.mnn` because reloading grouped mask
models per chunk costs much more than keeping those few sessions resident; it
still avoids caching legacy `mask_band_*.mnn` per-band heads.
The runner also honors `"time_batch"`, but any export that batches transformer
sequences must be quality-checked per model/backend before release.
The macOS CLI accepts `--profile` to print per-stage timing, including
`runSession`, input copy, output copy, and resize time for each MNN segment
category.

- macOS/iOS: use `CPU` for deterministic validation, `Metal` or `Auto` for
  acceleration when the linked MNN SDK includes Metal. Generic `Auto` precision
  uses `High` precision for `Metal` and `Auto` backends because normal precision
  can introduce accumulated error on deep transformer models.
- RoFormer `MetalFast` is the default when `precision=Auto`: it keeps the
  smallest validated `High` set and leaves the rest in `Normal` to minimize GPU
  memory and favor FP16 throughput on mobile. For native MNN `Attention`
  manifests exported as split `attention_ffn` segments, BSR-family `_attn`
  sessions run in `Normal` while `_ffn` stays `High`; MelBandRoFormer `_attn`
  sessions are forced to `High` because FP16 attention accumulates too much
  error end to end.
- Native MNN `Attention` segments require an MNN SDK built with
  `-DMNN_SUPPORT_TRANSFORMER_FUSE=ON`. The runtime sets
  `Interpreter::ATTENTION_OPTION=8` for RoFormer `layer_*` attention segments
  whose manifest has `"attention_op": "mnn"`. Unsplit native-attention
  transformer segments still run in `High` on Metal/Auto.
- Use `precision=Normal` with `precision_policy=Uniform` for the absolute
  lowest-memory run, `precision_policy=MetalAutocast` for a better quality/speed
  tradeoff, and `precision=High` for a full high-precision reference run.
- Use `segment_cache_policy=TransformersOnly` for mobile-first memory. Switch to
  `All` for maximum throughput; use `None` only when profiling a target backend.
- Android: use `CPU` for fallback, `OpenCL`, `Vulkan`, or `Auto` when the SDK and
  device support those backends.

## Current Coverage

The reusable C++ API covers RoFormer micro-segment inference end to end:
BS-RoFormer, Mel-Band-RoFormer, and HyperACE `segm` mask modes. It can also run
a fixed-shape full mask-core `.mnn` for presets with validated full-core export.
It performs WAV I/O, STFT/ISTFT, chunk overlap-add, MBR gather/scatter, MNN
segment/core execution, and stem reconstruction in C++.

`MNNModel` is model-agnostic and can execute fixed-shape single or multi tensor
`.mnn` cores such as VR or non-RoFormer cores when the caller supplies prepared
float tensors. Their model-specific audio DSP host pipelines are separate from
this RoFormer separator API.

Offline conversion remains in `tools/mnn_export` and may depend on
`pymss`/Torch/ONNX/PyMNN. Those dependencies are not part of the C++ runtime.
