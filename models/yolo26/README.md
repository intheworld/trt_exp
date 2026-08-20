# YOLO26n 模型

本目录保存 Ultralytics 官方 COCO 预训练检测模型：

- `yolo26n.pt`：原始 PyTorch 权重，SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- `yolo26n.onnx`：由 Ultralytics 8.4.122、PyTorch 2.11.0、ONNX opset 18 导出，固定输入 `1x3x640x640`，输出 `1x300x6`，SHA-256 `fbd29065229dca70386478c4a0139302e92b20cb133b05714eb5133ee57d09b0`

TensorRT plan 和 timing cache 与 TensorRT 版本、GPU 架构及构建配置相关，因此不纳入仓库。请按仓库根目录 README 的命令在已忽略的 `build/` 目录中生成。

模型来源及许可信息见 [Ultralytics YOLO26 官方文档](https://docs.ultralytics.com/models/yolo26/)。
