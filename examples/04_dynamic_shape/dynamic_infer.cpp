#include "trt_demo/common.hpp"

#include <NvInfer.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr << "用法: trt_dynamic_infer dynamic.plan height width\n";
        return 2;
    }

    try
    {
        int32_t const height = std::stoi(argv[2]);
        int32_t const width = std::stoi(argv[3]);
        trt_demo::require(height > 0 && width > 0, "H/W 必须为正数");
        trt_demo::Logger logger;
        auto engine = trt_demo::loadEngine(argv[1], logger);
        trt_demo::TrtUniquePtr<nvinfer1::IExecutionContext> context{
            engine->createExecutionContext()};
        trt_demo::require(static_cast<bool>(context), "创建 context 失败");
        trt_demo::require(context->setInputShape(
                              "input", nvinfer1::Dims4{1, 3, height, width}),
            "输入尺寸超出 optimization profile 范围");

        auto const inputDims = context->getTensorShape("input");
        auto const outputDims = context->getTensorShape("output");
        trt_demo::DeviceBuffer input(
            trt_demo::volume(inputDims) * sizeof(float));
        trt_demo::DeviceBuffer output(
            trt_demo::volume(outputDims) * sizeof(float));
        trt_demo::fillInput(input, nvinfer1::DataType::kFLOAT);
        trt_demo::require(context->setTensorAddress("input", input.data),
            "绑定 input 失败");
        trt_demo::require(context->setTensorAddress("output", output.data),
            "绑定 output 失败");

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

        std::cout << "输入 " << trt_demo::dimsToString(inputDims)
                  << " -> 输出 " << trt_demo::dimsToString(outputDims)
                  << "\noutput 前 10 个值:";
        trt_demo::printValues(output, nvinfer1::DataType::kFLOAT);
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}

