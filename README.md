# pymss-mnn

Standalone MNN inference runtime and conversion tools for music source separation models.

This repository no longer ships a Python/Torch inference path. Torch is kept only for offline model conversion under `tools/mnn_export/`. Runtime inference is C++/MNN under `runtime/mnn/`.

## Layout

```text
runtime/mnn/       C++ MNN runtime, CLI apps, CMake build, mobile-ready core
tools/mnn_export/  Offline Torch/ONNX/MNN conversion and validation tools
pymss/             Python model definitions and loaders used by conversion tools
```

## Install

For catalog lookup and model download only:

```sh
python -m pip install -e .
```

For users who want to convert checkpoints themselves:

```sh
python -m pip install -e ".[convert]"
```

The `convert` extra installs the Python-side conversion stack, including Torch,
ONNX Runtime, and PyMNN. It does not replace the MNN C++ SDK: C++ runtime builds
still need an MNN install with headers and libraries passed through `MNN_ROOT`,
and conversion commands need `MNNConvert` available on `PATH`.

## C++ Runtime

Build the macOS CLI apps:

```sh
cmake -S runtime/mnn -B runtime/mnn/build \
  -DMNN_ROOT=/path/to/MNN-install
cmake --build runtime/mnn/build --parallel 4
```

For iOS/Android integration, build the reusable core library without desktop apps:

```sh
cmake -S runtime/mnn -B runtime/mnn/build-mobile \
  -DMNN_ROOT=/path/to/MNN-install \
  -DMSS_MNN_BUILD_APPS=OFF
cmake --build runtime/mnn/build-mobile --parallel 4
```

Runtime code in `runtime/mnn/include`, `runtime/mnn/src`, and `runtime/mnn/apps` must not depend on Python, Torch, PyMNN, NumPy, librosa, or av.

## Conversion

Use `tools/mnn_export/` to convert source checkpoints to ONNX/MNN and validate exported cores. These tools may depend on Torch, NumPy, ONNX, ONNX Runtime, PyMNN, and the MNN converter because they are development-time tools, not runtime inference.

Examples:

```sh
python -m pip install -e ".[convert]"
export PATH=/path/to/MNN-install/bin:$PATH
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc
python tools/mnn_export/validate_expr_micro_segments.py --preset bsr_hyperace_voc --skip-torch
```

Model-specific runtime and validation notes live in:

- `runtime/mnn/docs/cpp_runtime.md`
- `runtime/mnn/docs/roformer_mask_core.md`
- `runtime/mnn/docs/non_roformer_mnn_validation.md`
- `runtime/mnn/docs/vr_mnn_validation.md`

## Catalog CLI

The `pymss` console script is only for model catalog lookup and downloading:

```sh
pymss list
pymss info bs_roformer_voc_hyperacev2
pymss download bs_roformer_voc_hyperacev2
```

Use MNN runtime apps for inference.
