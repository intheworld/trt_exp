#include "trt_demo/common.hpp"

#include <NvInfer.h>

#include <array>
#include <iostream>

namespace
{

float run(nvinfer1::ICudaEngine& engine)
{
    trt_demo::TrtUniquePtr<nvinfer1::IExecutionContext> context{
        engine.createExecutionContext()};
    trt_demo::require(static_cast<bool>(context), "创建 context 失败");
    std::array<float, 4> const inputHost{1.0F, 2.0F, 3.0F, 4.0F};
    trt_demo::DeviceBuffer input(sizeof(inputHost));
    trt_demo::DeviceBuffer output(sizeof(inputHost));
    trt_demo::checkCuda(cudaMemcpy(input.data, inputHost.data(), sizeof(inputHost),
                            cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D)");
    trt_demo::require(context->setTensorAddress("input", input.data),
        "绑定 input 失败");
    trt_demo::require(context->setTensorAddress("output", output.data),
        "绑定 output 失败");
    cudaStream_t stream{};
    trt_demo::checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    bool const enqueued = context->enqueueV3(stream);
    trt_demo::require(enqueued, "enqueueV3 失败");
    trt_demo::checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    trt_demo::checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    float first{};
    trt_demo::checkCuda(cudaMemcpy(&first, output.data, sizeof(first),
                            cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H)");
    return first;
}

} // namespace

int main()
{
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

        float const initialKernel = 1.0F;
        nvinfer1::Weights const kernel{
            nvinfer1::DataType::kFLOAT, &initialKernel, 1};
        nvinfer1::Weights const noBias{
            nvinfer1::DataType::kFLOAT, nullptr, 0};
        auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT,
            nvinfer1::Dims4{1, 1, 2, 2});
        auto* conv = network->addConvolutionNd(
            *input, 1, nvinfer1::DimsHW{1, 1}, kernel, noBias);
        trt_demo::require(input && conv, "创建 refit 网络失败");
        conv->setName("conv_1x1");
        conv->getOutput(0)->setName("output");
        network->markOutput(*conv->getOutput(0));
        trt_demo::require(network->setWeightsName(kernel, "conv_kernel"),
            "命名权重失败");
        trt_demo::require(network->markWeightsRefittable("conv_kernel"),
            "标记 refittable 权重失败");
        config->setFlag(nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL);

        trt_demo::TrtUniquePtr<nvinfer1::IHostMemory> plan{
            builder->buildSerializedNetwork(*network, *config)};
        trt_demo::require(static_cast<bool>(plan), "构建 refit engine 失败");
        trt_demo::TrtUniquePtr<nvinfer1::IRuntime> runtime{
            nvinfer1::createInferRuntime(logger)};
        trt_demo::TrtUniquePtr<nvinfer1::ICudaEngine> engine{
            runtime->deserializeCudaEngine(plan->data(), plan->size())};
        trt_demo::require(static_cast<bool>(engine), "反序列化失败");

        float const before = run(*engine);
        trt_demo::TrtUniquePtr<nvinfer1::IRefitter> refitter{
            nvinfer1::createInferRefitter(*engine, logger)};
        trt_demo::require(static_cast<bool>(refitter), "创建 refitter 失败");
        float const updatedKernel = 2.0F;
        nvinfer1::Weights const replacement{
            nvinfer1::DataType::kFLOAT, &updatedKernel, 1};
        trt_demo::require(refitter->setNamedWeights("conv_kernel", replacement),
            "设置新权重失败");
        trt_demo::require(refitter->refitCudaEngine(), "refitCudaEngine 失败");
        float const after = run(*engine);
        std::cout << "首个输出值: refit 前 " << before << "，refit 后 "
                  << after << "（期望 1 -> 2）\n";
        trt_demo::require(before == 1.0F && after == 2.0F,
            "refit 结果不符合预期");
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}

