#include "custom_layer_plugin.hpp"
#include "trt_demo/common.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstddef>
#include <iostream>
#include <string>

int main(int argc, char** argv)
try
{
    using namespace nvinfer1;
    using namespace trt_demo;
    if (argc < 2)
    {
        std::cerr << "usage: 07_plugin_builder custom_layer.onnx "
                     "[custom_layer.plan]\n";
        return 2;
    }
    ensureCustomLayerPluginRegistered();
    std::string const outputPath
        = argc > 2 ? argv[2] : "custom_layer.plan";
    Logger logger;
    TrtUniquePtr<IBuilder> builder{createInferBuilder(logger)};
    TrtUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};
    TrtUniquePtr<INetworkDefinition> network{builder->createNetworkV2(0U)};
    TrtUniquePtr<nvonnxparser::IParser> parser{
        nvonnxparser::createParser(*network, logger)};
    require(builder != nullptr && config != nullptr && network != nullptr
            && parser != nullptr,
        "failed to create TensorRT build objects");
    require(parser->parseFromFile(
                argv[1], static_cast<int32_t>(ILogger::Severity::kVERBOSE)),
        "custom-layer ONNX parse failed");
    config->setMemoryPoolLimit(
        MemoryPoolType::kWORKSPACE, std::size_t{1} << 28);
    TrtUniquePtr<IHostMemory> plan{
        builder->buildSerializedNetwork(*network, *config)};
    require(plan != nullptr, "buildSerializedNetwork failed");
    writeBinary(outputPath, plan->data(), plan->size());
    std::cout << "CustomLayer IPluginV3 engine saved to " << outputPath
              << '\n';
    return 0;
}
catch (std::exception const& error)
{
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
