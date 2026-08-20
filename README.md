# TensorRT 敲门即入门：可运行示例

本仓库整理自知乎专栏[《TensorRT 敲门即入门》](https://www.zhihu.com/column/c_1702267695536390144)的 13 篇文章。代码不是对旧 API 的机械复制，而是保留文章的学习路径，并适配当前环境的 TensorRT 11 命名张量 API、强类型网络与 `IPluginV3`。

已在以下环境实测：

- TensorRT 11.2.1.2
- CUDA Toolkit 12.5，NVIDIA Driver 580.173.02
- GCC 13.3、CMake 3.20+
- NVIDIA GeForce RTX 3080

## 示例目录

| 目录 | 可执行文件 | 学习内容 |
|---|---|---|
| `examples/00_network_api` | `trt_network_api` | 使用 C++ Network Definition API 创建最大池化网络并序列化 |
| `examples/01_onnx_builder` | `trt_onnx_builder` | 解析 ONNX、配置 workspace、构建 plan、保存/复用 timing cache |
| `examples/02_infer` | `trt_infer` | 反序列化、按名称分配 I/O、`setTensorAddress`、`enqueueV3` |
| `examples/03_mixed_precision` | `trt_precision` | FP32、FP16 和 TensorRT 11 显式 INT8 Q/DQ |
| `examples/04_dynamic_shape` | `trt_dynamic_build`、`trt_dynamic_infer` | optimization profile、运行时 `setInputShape` |
| `examples/05_refit` | `trt_refit` | 命名权重、构建可 refit engine、运行时替换权重 |
| `examples/07_custom_plugin` | `trt_plugin_build`、`trt_plugin_infer` | ONNX 自定义节点、`IPluginV3`、CUDA kernel、序列化与注册 |
| `examples/08_yolo26` | `trt_yolo26` | YOLO26n 图片预处理、端到端无 NMS 检测、坐标还原与结果绘制，含测试图片及参考结果 |
| `models` | `make_models.py` | 生成静态、动态及自定义算子 ONNX 模型 |
| `models/yolo26` | `yolo26n.pt`、`yolo26n.onnx` | YOLO26n 官方 COCO 权重及固定 `640x640` ONNX |

文章与示例的逐篇对应关系见 [docs/article_map.md](docs/article_map.md)。仓库根目录原有的 `trt_simple.py` 和 `trt_simple.cpp` 仍保留，作为最小构建示例。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

CMake 会查找 TensorRT 的 `nvinfer`、`nvonnxparser`，以及 CUDA Toolkit。Debug 构建便于在 `gdb` 中下断点；需要性能测试时改用 `-DCMAKE_BUILD_TYPE=Release`。

## 快速运行

### 1. Network API 构建与推理

```bash
./build/trt_network_api network_api.plan
./build/trt_infer network_api.plan
```

输入由示例自动填充，推理程序会打印 I/O 形状和输出前 10 个数值。

### 2. ONNX 解析、构建与 timing cache

先安装仅用于生成测试模型的 Python 依赖：

```bash
python3 -m pip install -r requirements.txt
python3 models/make_models.py generated_models
```

然后构建并推理：

```bash
./build/trt_onnx_builder \
  generated_models/static_pool.onnx static_pool.plan static_pool.cache
./build/trt_infer static_pool.plan
```

再次执行 builder 时会复用 `static_pool.cache`。这对应旧文章的算法选择主题；TensorRT 11 已移除 `IAlgorithmSelector`，推荐使用可序列化 timing cache。

### 3. 混合精度

```bash
./build/trt_precision fp32 fp32.plan
./build/trt_precision fp16 fp16.plan
./build/trt_precision int8 int8_qdq.plan

./build/trt_infer fp32.plan
./build/trt_infer fp16.plan
./build/trt_infer int8_qdq.plan
```

TensorRT 11 不再提供文章中的隐式 INT8 校准器和 `BuilderFlag::kFP16/kINT8`。本示例中 FP16 由张量类型明确表达；INT8 使用 `Quantize -> Dequantize` 层显式表达 scale，因而不再需要 calibration cache。

### 4. Dynamic Shape

```bash
./build/trt_dynamic_build dynamic.plan
./build/trt_dynamic_infer dynamic.plan 32 48
./build/trt_dynamic_infer dynamic.plan 96 128
```

示例允许高度和宽度位于 `[8, 128]`，优化尺寸为 `32x32`。超出 profile 范围会明确报错。

### 5. Refit

```bash
./build/trt_refit
```

程序创建一个 `1x1` 卷积，首次推理权重为 `1`；随后通过 `IRefitter` 将权重改成 `2`，同一 engine 的首个输出由 `1` 变为 `2`。

### 6. 自定义算子

文章算子公式为：

```text
output = (input + offset) * multiplier * alpha
```

运行完整流程：

```bash
python3 models/make_models.py generated_models
./build/trt_plugin_build \
  generated_models/custom_layer.onnx custom_layer.plan
./build/trt_plugin_infer custom_layer.plan
```

`trt_plugin_infer` 会计算参考结果并检查最大绝对误差。实现使用传入的 CUDA stream，并根据张量元素数量自动计算 grid，不再把 kernel 固定为单 block、24 个线程。

### 7. YOLO26n 目标检测

[YOLO26](https://docs.ultralytics.com/models/yolo26/) 是 Ultralytics 于 2026 年发布的端到端实时视觉模型。默认 one-to-one 检测头输出 `(N, 300, 6)`，每条结果依次为 `x1, y1, x2, y2, score, class_id`，不需要额外执行 NMS。仓库包含 PyTorch、ONNX、测试图片及参考输出；与本机环境绑定的 TensorRT plan 和 timing cache 不纳入仓库。

如需从官方权重重新导出 ONNX，可使用临时 Python 环境，不会向仓库添加中间脚本：

```bash
python3 -m venv build/yolo26-export-venv
build/yolo26-export-venv/bin/pip install -U ultralytics onnx onnxslim
cd models/yolo26
../../build/yolo26-export-venv/bin/yolo export model=yolo26n.pt \
  format=onnx imgsz=640 batch=1 dynamic=False simplify=True
cd ../..
```

构建 TensorRT plan。YOLO26n 默认端到端输出已经完成 top-k 选择，因此不要使用 `end2end=False` 导出：

```bash
./build/trt_onnx_builder \
  models/yolo26/yolo26n.onnx \
  build/yolo26n.plan build/yolo26n.cache
```

`build/` 已被 `.gitignore` 忽略。切换 TensorRT 版本或 GPU 后必须重新执行构建命令。

对任意 JPEG/PNG 图片执行检测并保存画框结果：

```bash
./build/trt_yolo26 \
  build/yolo26n.plan \
  examples/08_yolo26/assets/bus.jpg build/yolo26-result.jpg
# 可选的第五个参数是置信度阈值
./build/trt_yolo26 \
  build/yolo26n.plan input.jpg output.jpg 0.4
```

仓库内的参考画框结果位于 `examples/08_yolo26/assets/bus-result.jpg`。

示例在 C++ 中完整实现 Ultralytics 的等比例缩放和灰边填充、BGR 到 RGB、NCHW 排列、`[0, 1]` 归一化、COCO 类别解析、原图坐标还原及绘图。若系统没有 OpenCV 开发包，CMake 会跳过 `trt_yolo26`。

## 调试建议

查看 TensorRT 网络或 plan：

```bash
trtexec --onnx=generated_models/static_pool.onnx --verbose
trtexec --loadEngine=static_pool.plan --dumpLayerInfo --profilingVerbosity=detailed
```

调试自定义 kernel：

```bash
compute-sanitizer ./build/trt_plugin_infer custom_layer.plan
```

engine 与 TensorRT 版本、GPU 架构和构建配置相关，不应提交到版本库；切换环境后应重新构建。
