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
options.threads = 1;

mss_mnn::RoformerSeparator separator(options);
mss_mnn::AudioBuffer input = mss_mnn::read_wav("input.wav");
std::vector<mss_mnn::AudioBuffer> stems = separator.separate(input);
```

For mobile apps, feed `AudioBuffer` directly from the platform audio decoder and
write platform-native output. The WAV helpers are only a portable CLI/testing
convenience.

## Backends

`MNNBackend` supports `CPU`, `Auto`, `Metal`, `OpenCL`, and `Vulkan`.

- macOS/iOS: use `CPU` for deterministic validation, `Metal` or `Auto` for
  acceleration when the linked MNN SDK includes Metal.
- Android: use `CPU` for fallback, `OpenCL`, `Vulkan`, or `Auto` when the SDK and
  device support those backends.

## Current Coverage

The reusable C++ API covers RoFormer micro-segment inference end to end:
BS-RoFormer, Mel-Band-RoFormer, and HyperACE `segm` mask modes. It performs WAV
I/O, STFT/ISTFT, chunk overlap-add, MBR gather/scatter, MNN segment execution,
and stem reconstruction in C++.

`MNNModel` is model-agnostic and can execute fixed-shape single or multi tensor
`.mnn` cores such as VR or non-RoFormer cores when the caller supplies prepared
float tensors. Their model-specific audio DSP host pipelines are separate from
this RoFormer separator API.

Offline conversion remains in `tools/mnn_export` and may depend on
`pymss`/Torch/ONNX/PyMNN. Those dependencies are not part of the C++ runtime.
