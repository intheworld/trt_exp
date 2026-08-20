#include "trt_demo/common.hpp"

#include <NvInfer.h>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "用法: trt_infer model.plan\n";
        return 2;
    }

    try
    {
        trt_demo::Logger logger;
        auto engine = trt_demo::loadEngine(argv[1], logger);
        trt_demo::TrtUniquePtr<nvinfer1::IExecutionContext> context{
            engine->createExecutionContext()};
        trt_demo::require(static_cast<bool>(context), "创建 context 失败");

        std::vector<trt_demo::DeviceBuffer> buffers;
        buffers.reserve(static_cast<std::size_t>(engine->getNbIOTensors()));
        for (int32_t index = 0; index < engine->getNbIOTensors(); ++index)
        {
            char const* name = engine->getIOTensorName(index);
            auto const mode = engine->getTensorIOMode(name);
            auto const dims = context->getTensorShape(name);
            auto const type = engine->getTensorDataType(name);
            std::size_t const bytes
                = trt_demo::volume(dims) * trt_demo::dataTypeSize(type);
            buffers.emplace_back(bytes);
            trt_demo::require(context->setTensorAddress(name, buffers.back().data),
                std::string("绑定张量地址失败: ") + name);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
            {
                trt_demo::fillInput(buffers.back(), type);
            }
            std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "输入 " : "输出 ")
                      << name << ' ' << trt_demo::dimsToString(dims)
                      << "，" << bytes << " bytes\n";
        }

        cudaStream_t stream{};
        trt_demo::checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        bool const enqueued = context->enqueueV3(stream);
        if (!enqueued)
        {
            cudaStreamDestroy(stream);
            throw std::runtime_error("enqueueV3 失败");
        }
        trt_demo::checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        trt_demo::checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

        for (int32_t index = 0; index < engine->getNbIOTensors(); ++index)
        {
            char const* name = engine->getIOTensorName(index);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
            {
                std::cout << name << " 前 10 个值:";
                trt_demo::printValues(
                    buffers[static_cast<std::size_t>(index)],
                    engine->getTensorDataType(name));
            }
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}
