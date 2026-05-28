# pymss-mnn

用于音乐源分离模型的 MNN 推理运行时和离线转换工具。

本仓库不再提供 Python/Torch 推理路径。Torch 只保留在 `tools/mnn_export/` 中用于离线模型转换；真正推理路径在 `runtime/mnn/`，使用 C++ 和 MNN Runtime。

## 目录

```text
runtime/mnn/       C++ MNN runtime、CLI apps、CMake 构建和移动端共用核心
tools/mnn_export/  离线 Torch/ONNX/MNN 转换与验证工具
pymss/             转换工具需要的 Python 模型定义和加载器
```

## 安装

只查询 catalog 和下载模型：

```sh
python -m pip install -e .
```

需要自行转换 checkpoint 的用户安装转换依赖：

```sh
python -m pip install -e ".[convert]"
```

`convert` extra 会安装 Python 侧转换栈，包括 Torch、ONNX Runtime 和 PyMNN。它不替代 MNN C++ SDK：构建 C++ runtime 仍然需要带 headers/libs 的 MNN install，并通过 `MNN_ROOT` 传入；转换命令也需要 `MNNConvert` 在 `PATH` 中可用。

## C++ Runtime

构建 macOS CLI：

```sh
cmake -S runtime/mnn -B runtime/mnn/build \
  -DMNN_ROOT=/path/to/MNN-install
cmake --build runtime/mnn/build --parallel 4
```

iOS/Android 集成时可以只构建核心库，不构建桌面 CLI：

```sh
cmake -S runtime/mnn -B runtime/mnn/build-mobile \
  -DMNN_ROOT=/path/to/MNN-install \
  -DMSS_MNN_BUILD_APPS=OFF
cmake --build runtime/mnn/build-mobile --parallel 4
```

`runtime/mnn/include`、`runtime/mnn/src`、`runtime/mnn/apps` 中的运行时代码不能依赖 Python、Torch、PyMNN、NumPy、librosa 或 av。

## 模型转换

使用 `tools/mnn_export/` 将源 checkpoint 转成 ONNX/MNN，并验证导出的神经网络核心。这些工具可以依赖 Torch、NumPy、ONNX、ONNX Runtime、PyMNN 和 MNN converter，因为它们只用于开发期离线转换，不进入运行时推理。

示例：

```sh
python -m pip install -e ".[convert]"
export PATH=/path/to/MNN-install/bin:$PATH
python tools/mnn_export/export_expr_micro_segments.py --preset bsr_hyperace_voc
python tools/mnn_export/validate_expr_micro_segments.py --preset bsr_hyperace_voc --skip-torch
```

模型相关运行时和验证说明：

- `runtime/mnn/docs/cpp_runtime.md`
- `runtime/mnn/docs/roformer_mask_core.md`
- `runtime/mnn/docs/non_roformer_mnn_validation.md`
- `runtime/mnn/docs/vr_mnn_validation.md`

## Catalog CLI

`pymss` 命令只用于模型 catalog 查询和下载：

```sh
pymss list
pymss info bs_roformer_voc_hyperacev2
pymss download bs_roformer_voc_hyperacev2
```

推理请使用 MNN runtime apps。
