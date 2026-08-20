#include "trt_demo/common.hpp"

#include <NvInfer.h>

#include <iostream>

int main(int argc, char* argv[])
{
    std::string const outputPath
        = argc > 1 ? argv[1] : "dynamic.plan";
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
        trt_demo::require(network && config, "创建 network/config 失败");

        auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims4{1, 3, -1, -1});
        trt_demo::require(input != nullptr, "添加动态输入失败");
        auto* pool = network->addPoolingNd(
            *input, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
        trt_demo::require(pool != nullptr, "添加池化层失败");
        pool->setStrideNd(nvinfer1::DimsHW{2, 2});
        pool->getOutput(0)->setName("output");
        network->markOutput(*pool->getOutput(0));

        auto* profile = builder->createOptimizationProfile();
        trt_demo::require(profile != nullptr, "创建 optimization profile 失败");
        trt_demo::require(profile->setDimensions("input",
                              nvinfer1::OptProfileSelector::kMIN,
                              nvinfer1::Dims4{1, 3, 8, 8}),
            "设置最小尺寸失败");
        trt_demo::require(profile->setDimensions("input",
                              nvinfer1::OptProfileSelector::kOPT,
                              nvinfer1::Dims4{1, 3, 32, 32}),
            "设置最优尺寸失败");
        trt_demo::require(profile->setDimensions("input",
                              nvinfer1::OptProfileSelector::kMAX,
                              nvinfer1::Dims4{1, 3, 128, 128}),
            "设置最大尺寸失败");
        trt_demo::require(config->addOptimizationProfile(profile) >= 0,
            "添加 optimization profile 失败");
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE, std::size_t{256} << 20);

        trt_demo::TrtUniquePtr<nvinfer1::IHostMemory> plan{
            builder->buildSerializedNetwork(*network, *config)};
        trt_demo::require(static_cast<bool>(plan), "构建动态 engine 失败");
        trt_demo::writeBinary(outputPath, plan->data(), plan->size());
        std::cout << "已生成 " << outputPath
                  << "，允许 H/W 范围 [8, 128]，优化尺寸 32x32\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}

