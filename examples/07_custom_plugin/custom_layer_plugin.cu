#include "custom_layer_plugin.hpp"

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <new>

namespace
{

constexpr char kPluginName[] = "CustomLayer";
constexpr char kPluginVersion[] = "1";
constexpr char kPluginNamespace[] = "";

__global__ void customLayerKernel(float const* input, float const* offset,
    float const* multiplier, float alpha, float* output, std::int64_t count)
{
    std::int64_t const index
        = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        output[index]
            = (input[index] + offset[index]) * multiplier[index] * alpha;
    }
}

std::int64_t tensorVolume(nvinfer1::Dims const& dims)
{
    std::int64_t result{1};
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            return -1;
        }
        result *= dims.d[i];
    }
    return result;
}

bool sameShape(nvinfer1::Dims const& left, nvinfer1::Dims const& right)
{
    if (left.nbDims != right.nbDims)
    {
        return false;
    }
    for (int32_t i = 0; i < left.nbDims; ++i)
    {
        if (left.d[i] != right.d[i])
        {
            return false;
        }
    }
    return true;
}

} // namespace

CustomLayerPlugin::CustomLayerPlugin(float alpha) noexcept
    : alpha_(alpha)
{
    resetSerializationFields();
}

CustomLayerPlugin::CustomLayerPlugin(
    CustomLayerPlugin const& other) noexcept
    : alpha_(other.alpha_)
{
    resetSerializationFields();
}

void CustomLayerPlugin::resetSerializationFields() noexcept
{
    serializedAlpha_ = nvinfer1::PluginField{
        "alpha", &alpha_, nvinfer1::PluginFieldType::kFLOAT32, 1};
    serializedFields_.nbFields = 1;
    serializedFields_.fields = &serializedAlpha_;
}

nvinfer1::IPluginCapability* CustomLayerPlugin::getCapabilityInterface(
    nvinfer1::PluginCapabilityType type) noexcept
{
    switch (type)
    {
    case nvinfer1::PluginCapabilityType::kCORE:
        return static_cast<nvinfer1::IPluginV3OneCore*>(this);
    case nvinfer1::PluginCapabilityType::kBUILD:
        return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
    case nvinfer1::PluginCapabilityType::kRUNTIME:
        return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* CustomLayerPlugin::clone() noexcept
{
    return new (std::nothrow) CustomLayerPlugin(*this);
}

nvinfer1::AsciiChar const* CustomLayerPlugin::getPluginName() const noexcept
{
    return kPluginName;
}

nvinfer1::AsciiChar const* CustomLayerPlugin::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

nvinfer1::AsciiChar const* CustomLayerPlugin::getPluginNamespace() const noexcept
{
    return kPluginNamespace;
}

int32_t CustomLayerPlugin::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const*, int32_t nbInputs,
    nvinfer1::DynamicPluginTensorDesc const*, int32_t nbOutputs) noexcept
{
    return nbInputs == 3 && nbOutputs == 1 ? 0 : -1;
}

int32_t CustomLayerPlugin::getOutputDataTypes(
    nvinfer1::DataType* outputTypes, int32_t nbOutputs,
    nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    if (nbInputs != 3 || nbOutputs != 1)
    {
        return -1;
    }
    outputTypes[0] = inputTypes[0];
    return 0;
}

int32_t CustomLayerPlugin::getOutputShapes(
    nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
    nvinfer1::DimsExprs const*, int32_t nbShapeInputs,
    nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
    nvinfer1::IExprBuilder&) noexcept
{
    if (nbInputs != 3 || nbShapeInputs != 0 || nbOutputs != 1)
    {
        return -1;
    }
    outputs[0] = inputs[0];
    return 0;
}

bool CustomLayerPlugin::supportsFormatCombination(int32_t pos,
    nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
    int32_t nbOutputs) noexcept
{
    if (nbInputs != 3 || nbOutputs != 1 || pos < 0 || pos >= 4)
    {
        return false;
    }
    auto const& desc = inOut[pos].desc;
    return desc.type == nvinfer1::DataType::kFLOAT
        && desc.format == nvinfer1::TensorFormat::kLINEAR;
}

int32_t CustomLayerPlugin::getNbOutputs() const noexcept
{
    return 1;
}

std::size_t CustomLayerPlugin::getWorkspaceSize(
    nvinfer1::DynamicPluginTensorDesc const*, int32_t,
    nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t CustomLayerPlugin::onShapeChange(
    nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != 3 || nbOutputs != 1)
    {
        return -1;
    }
    return sameShape(inputs[0].dims, inputs[1].dims)
            && sameShape(inputs[0].dims, inputs[2].dims)
            && sameShape(inputs[0].dims, outputs[0].dims)
        ? 0
        : -1;
}

int32_t CustomLayerPlugin::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const*, void const* const* inputs,
    void* const* outputs, void*, cudaStream_t stream) noexcept
{
    std::int64_t const count = tensorVolume(inputDesc[0].dims);
    if (count < 0)
    {
        return -1;
    }
    int32_t constexpr threads{256};
    int32_t const blocks
        = static_cast<int32_t>((count + threads - 1) / threads);
    customLayerKernel<<<blocks, threads, 0, stream>>>(
        static_cast<float const*>(inputs[0]),
        static_cast<float const*>(inputs[1]),
        static_cast<float const*>(inputs[2]), alpha_,
        static_cast<float*>(outputs[0]), count);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -1;
}

nvinfer1::IPluginV3* CustomLayerPlugin::attachToContext(
    nvinfer1::IPluginResourceContext*) noexcept
{
    return clone();
}

nvinfer1::PluginFieldCollection const*
CustomLayerPlugin::getFieldsToSerialize() noexcept
{
    return &serializedFields_;
}

CustomLayerPluginCreator::CustomLayerPluginCreator() noexcept
{
    attributes_.emplace_back(
        "alpha", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1);
    fieldNames_.nbFields = static_cast<int32_t>(attributes_.size());
    fieldNames_.fields = attributes_.data();
}

nvinfer1::IPluginV3* CustomLayerPluginCreator::createPlugin(
    nvinfer1::AsciiChar const*, nvinfer1::PluginFieldCollection const* fields,
    nvinfer1::TensorRTPhase) noexcept
{
    float alpha{1.0F};
    if (fields != nullptr)
    {
        for (int32_t i = 0; i < fields->nbFields; ++i)
        {
            auto const& field = fields->fields[i];
            if (field.name != nullptr && std::strcmp(field.name, "alpha") == 0
                && field.data != nullptr)
            {
                alpha = *static_cast<float const*>(field.data);
            }
        }
    }
    return new (std::nothrow) CustomLayerPlugin(alpha);
}

nvinfer1::PluginFieldCollection const*
CustomLayerPluginCreator::getFieldNames() noexcept
{
    return &fieldNames_;
}

nvinfer1::AsciiChar const*
CustomLayerPluginCreator::getPluginName() const noexcept
{
    return kPluginName;
}

nvinfer1::AsciiChar const*
CustomLayerPluginCreator::getPluginVersion() const noexcept
{
    return kPluginVersion;
}

nvinfer1::AsciiChar const*
CustomLayerPluginCreator::getPluginNamespace() const noexcept
{
    return kPluginNamespace;
}

REGISTER_TENSORRT_PLUGIN(CustomLayerPluginCreator);

void ensureCustomLayerPluginRegistered() noexcept
{
}
