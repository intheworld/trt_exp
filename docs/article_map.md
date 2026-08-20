# 专栏文章与仓库代码映射

本页用于核对专栏 13 篇文章中的代码示例是否已覆盖。文章发布时使用的是旧版 TensorRT，本仓库统一采用 TensorRT 11.2 可编译、可运行的写法。

## 学习笔记

| 文章 | 原文示例 | 仓库对应实现 |
|---|---|---|
| [学习笔记（一）-- 简介](https://zhuanlan.zhihu.com/p/662992072) | Builder、ONNX parser、序列化；Runtime、Context、`enqueueV2` | `00_network_api`、`01_onnx_builder`、`02_infer` |
| [学习笔记（二）-- 混合精度](https://zhuanlan.zhihu.com/p/663000989) | FP16/INT8 flag、动态范围、calibrator、显式精度 | `03_mixed_precision`；旧隐式量化迁移为显式 Q/DQ |
| [学习笔记（三）-- Dynamic Shape](https://zhuanlan.zhihu.com/p/663073342) | `-1` 动态维度、optimization profile、运行时设置尺寸 | `04_dynamic_shape` |
| [学习笔记（四）-- 编程接口](https://zhuanlan.zhihu.com/p/663291005) | Network API、卷积、标记输出、Refitter、算法选择 | `00_network_api`、`05_refit`、`01_onnx_builder` 的 timing cache |

## 实战系列

| 文章 | 仓库对应实现 |
|---|---|
| [实战（一）-- Builder](https://zhuanlan.zhihu.com/p/665645167) | `examples/01_onnx_builder/onnx_builder.cpp` |
| [实战（二）-- Infer](https://zhuanlan.zhihu.com/p/665858856) | `examples/02_infer/infer.cpp` |
| [实战（三）-- 混合精度（上）](https://zhuanlan.zhihu.com/p/666908705) | `examples/03_mixed_precision/mixed_precision.cpp` |
| [实战（四）-- 混合精度（下）](https://zhuanlan.zhihu.com/p/667098156) | 显式 Q/DQ 示例及 README 的迁移说明；旧 calibrator 在 TensorRT 11 中已删除 |
| [实战（五）-- dynamic shape（上）](https://zhuanlan.zhihu.com/p/668110979) | `examples/04_dynamic_shape/dynamic_build.cpp` |
| [实战（六）-- dynamic shape（下）](https://zhuanlan.zhihu.com/p/668120127) | `examples/04_dynamic_shape/dynamic_infer.cpp` |
| [实战（七）-- 自定义算子（上）](https://zhuanlan.zhihu.com/p/669382709) | `models/make_models.py`、`examples/07_custom_plugin/infer.cpp` |
| [实战（八）-- 自定义算子（中）](https://zhuanlan.zhihu.com/p/669389426) | `examples/07_custom_plugin/build.cpp` |
| [实战（九）-- 自定义算子（下）](https://zhuanlan.zhihu.com/p/669395745) | `custom_layer_plugin.hpp/.cu`，接口升级为 `IPluginV3` |

## TensorRT 11 迁移要点

| 文章中的旧写法 | 本仓库写法 |
|---|---|
| `kEXPLICIT_BATCH` / `kEXPLICIT_PRECISION` | TensorRT 11 网络默认显式 batch、强类型，`createNetworkV2(0U)` |
| `setMaxWorkspaceSize` | `setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, ...)` |
| `buildEngineWithConfig` 后再 `serialize()` | `buildSerializedNetwork()` |
| 手工调用 `destroy()` | `std::unique_ptr` RAII |
| binding index、`getNbBindings()` | 张量名称、`getNbIOTensors()`、`getIOTensorName()` |
| `setBindingDimensions()` | `setInputShape()` |
| `enqueueV2(bindings, ...)` | `setTensorAddress()` 后调用 `enqueueV3()` |
| `BuilderFlag::kFP16/kINT8`、calibrator | 强类型 FP16、显式 Quantize/Dequantize |
| `IAlgorithmSelector` | 可序列化 `ITimingCache` |
| `IPluginV2IOExt` | `IPluginV3OneCore/Build/Runtime` |

专栏中的 calibration table 文本是旧隐式 INT8 校准结果示意，不是独立程序，因此没有复制成无效的数据文件；相应概念由 `03_mixed_precision` 的显式 scale 展示。
