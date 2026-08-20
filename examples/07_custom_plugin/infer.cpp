#include "custom_layer_plugin.hpp"
#include "trt_demo/common.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main(int argc, char** argv)
try
{
    using namespace nvinfer1;
    using namespace trt_demo;
    if (argc < 2)
    {
        std::cerr << "usage: 07_plugin_infer custom_layer.plan\n";
        return 2;
    }
    ensureCustomLayerPluginRegistered();
    Logger logger;
    auto engine = loadEngine(argv[1], logger);
    TrtUniquePtr<IExecutionContext> context{engine->createExecutionContext()};
    require(context != nullptr, "createExecutionContext failed");

    std::vector<std::string> names;
    std::vector<DeviceBuffer> buffers;
    for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
    {
        char const* name = engine->getIOTensorName(i);
        auto const dims = context->getTensorShape(name);
        auto const count = volume(dims);
        require(engine->getTensorDataType(name) == DataType::kFLOAT,
            "plugin example expects FP32 I/O");
        names.emplace_back(name);
        buffers.emplace_back(count * sizeof(float));
        require(context->setTensorAddress(name, buffers.back().data),
            "setTensorAddress failed");
        if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT)
        {
            std::vector<float> input(static_cast<std::size_t>(count));
            for (std::size_t index = 0; index < input.size(); ++index)
            {
                input[index] = static_cast<float>(index);
            }
            checkCuda(cudaMemcpy(buffers.back().data, input.data(),
                          input.size() * sizeof(float), cudaMemcpyHostToDevice),
                "cudaMemcpy input");
        }
    }

    cudaStream_t stream{};
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    require(context->enqueueV3(stream), "enqueueV3 failed");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

    for (std::size_t i = 0; i < names.size(); ++i)
    {
        if (engine->getTensorIOMode(names[i].c_str()) == TensorIOMode::kOUTPUT)
        {
            std::cout << "output = (input + 1) * square(index) * 0.1\n";
            printValues(buffers[i], DataType::kFLOAT, 24);
            std::size_t const count = buffers[i].bytes / sizeof(float);
            std::vector<float> output(count);
            checkCuda(cudaMemcpy(output.data(), buffers[i].data,
                          buffers[i].bytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy output");
            float maxError{0.0F};
            for (std::size_t index = 0; index < count; ++index)
            {
                float const expected = (static_cast<float>(index) + 1.0F)
                    * static_cast<float>(index * index) * 0.1F;
                maxError = std::max(
                    maxError, std::abs(output[index] - expected));
            }
            std::cout << "max absolute error: " << maxError << '\n';
            require(maxError < 2.0e-4F, "CustomLayer result mismatch");
        }
    }
    return 0;
}
catch (std::exception const& error)
{
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
