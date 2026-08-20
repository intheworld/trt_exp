#include "trt_demo/common.hpp"

#include <NvInfer.h>

#include <iostream>
#include <string>

namespace
{

enum class Precision
{
    kFp32,
    kFp16,
    kInt8Qdq,
};

Precision parsePrecision(std::string const& value)
{
    if (value == "fp32") return Precision::kFp32;
    if (value == "fp16") return Precision::kFp16;
    if (value == "int8") return Precision::kInt8Qdq;
    throw std::runtime_error("精度必须是 fp32、fp16 或 int8");
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "用法: trt_precision fp32|fp16|int8 output.plan\n";
        return 2;
    }

    try
    {
        Precision const precision = parsePrecision(argv[1]);
        trt_demo::Logger logger;
        trt_demo::TrtUniquePtr<nvinfer1::IBuilder> builder{
            nvinfer1::createInferBuilder(logger)};
        trt_demo::require(static_cast<bool>(builder), "创建 builder 失败");
        trt_demo::TrtUniquePtr<nvinfer1::INetworkDefinition> network{
            builder->createNetworkV2(0U)};
        trt_demo::TrtUniquePtr<nvinfer1::IBuilderConfig> config{
            builder->createBuilderConfig()};
        trt_demo::require(network && config, "创建 network/config 失败");

        nvinfer1::DataType const inputType = precision == Precision::kFp16
            ? nvinfer1::DataType::kHALF
            : nvinfer1::DataType::kFLOAT;
        auto* input = network->addInput(
            "input", inputType, nvinfer1::Dims4{1, 3, 8, 8});
        trt_demo::require(input != nullptr, "添加输入失败");

        nvinfer1::ITensor* poolInput = input;
        float const scaleValue = 1.0F / 127.0F;
        nvinfer1::Weights const scaleWeights{
            nvinfer1::DataType::kFLOAT, &scaleValue, 1};
        if (precision == Precision::kInt8Qdq)
        {
            nvinfer1::Dims scalar{};
            scalar.nbDims = 0;
            auto* scale = network->addConstant(scalar, scaleWeights);
            trt_demo::require(scale != nullptr, "添加量化 scale 失败");
            auto* quantize = network->addQuantize(
                *input, *scale->getOutput(0), nvinfer1::DataType::kINT8);
            trt_demo::require(quantize != nullptr, "添加 Quantize 层失败");
            quantize->setName("explicit_quantize");
            auto* dequantize = network->addDequantize(*quantize->getOutput(0),
                *scale->getOutput(0), nvinfer1::DataType::kFLOAT);
            trt_demo::require(dequantize != nullptr, "添加 Dequantize 层失败");
            dequantize->setName("explicit_dequantize");
            poolInput = dequantize->getOutput(0);
        }

        auto* pool = network->addPoolingNd(*poolInput,
            nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
        trt_demo::require(pool != nullptr, "添加池化层失败");
        pool->setStrideNd(nvinfer1::DimsHW{2, 2});
        pool->getOutput(0)->setName("output");
        network->markOutput(*pool->getOutput(0));

        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE, std::size_t{256} << 20);
        trt_demo::TrtUniquePtr<nvinfer1::IHostMemory> plan{
            builder->buildSerializedNetwork(*network, *config)};
        trt_demo::require(static_cast<bool>(plan), "构建 engine 失败");
        trt_demo::writeBinary(argv[2], plan->data(), plan->size());
        std::cout << "已生成 " << argv[2] << "。TensorRT 11 使用强类型网络；"
                  << "int8 示例采用显式 Q/DQ。\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}

