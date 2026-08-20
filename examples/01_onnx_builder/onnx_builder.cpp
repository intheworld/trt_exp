#include "trt_demo/common.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr << "用法: trt_onnx_builder model.onnx model.plan [timing.cache]\n";
        return 2;
    }

    try
    {
        trt_demo::Logger logger;
        trt_demo::TrtUniquePtr<nvinfer1::IBuilder> builder{
            nvinfer1::createInferBuilder(logger)};
        trt_demo::require(static_cast<bool>(builder), "创建 builder 失败");
        trt_demo::TrtUniquePtr<nvinfer1::INetworkDefinition> network{
            builder->createNetworkV2(0U)};
        trt_demo::TrtUniquePtr<nvinfer1::IBuilderConfig> config{
            builder->createBuilderConfig()};
        trt_demo::TrtUniquePtr<nvonnxparser::IParser> parser{
            nvonnxparser::createParser(*network, logger)};
        trt_demo::require(network && config && parser, "创建构建对象失败");

        trt_demo::require(parser->parseFromFile(argv[1],
                              static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)),
            "ONNX 解析失败");
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE, std::size_t{1} << 30);

        trt_demo::TrtUniquePtr<nvinfer1::ITimingCache> timingCache;
        if (argc > 3)
        {
            std::vector<char> cacheBytes;
            std::ifstream cacheInput(argv[3], std::ios::binary);
            if (cacheInput)
            {
                cacheBytes = trt_demo::readBinary(argv[3]);
            }
            timingCache.reset(config->createTimingCache(
                cacheBytes.empty() ? nullptr : cacheBytes.data(), cacheBytes.size()));
            trt_demo::require(static_cast<bool>(timingCache), "创建 timing cache 失败");
            trt_demo::require(config->setTimingCache(*timingCache, false),
                "设置 timing cache 失败");
        }

        std::cout << "网络输入 " << network->getNbInputs() << "，输出 "
                  << network->getNbOutputs() << '\n';
        trt_demo::TrtUniquePtr<nvinfer1::IHostMemory> plan{
            builder->buildSerializedNetwork(*network, *config)};
        trt_demo::require(static_cast<bool>(plan), "构建 engine 失败");
        trt_demo::writeBinary(argv[2], plan->data(), plan->size());

        if (timingCache)
        {
            trt_demo::TrtUniquePtr<nvinfer1::IHostMemory> serialized{
                timingCache->serialize()};
            trt_demo::writeBinary(argv[3], serialized->data(), serialized->size());
        }
        std::cout << "已生成 " << argv[2] << '\n';
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}

