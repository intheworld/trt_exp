#pragma once

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_demo
{

class Logger final : public nvinfer1::ILogger
{
public:
    explicit Logger(Severity threshold = Severity::kINFO)
        : threshold_(threshold)
    {
    }

    void log(Severity severity, char const* message) noexcept override
    {
        if (severity <= threshold_)
        {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }

private:
    Severity threshold_;
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

inline void require(bool condition, std::string const& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

inline void checkCuda(cudaError_t status, char const* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

inline std::vector<char> readBinary(std::string const& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(static_cast<bool>(input), "无法打开文件: " + path);
    auto const size = input.tellg();
    require(size >= 0, "无法读取文件大小: " + path);
    std::vector<char> data(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(data.data(), static_cast<std::streamsize>(data.size()));
    require(static_cast<bool>(input), "读取文件失败: " + path);
    return data;
}

inline void writeBinary(
    std::string const& path, void const* data, std::size_t size)
{
    std::ofstream output(path, std::ios::binary);
    require(static_cast<bool>(output), "无法创建文件: " + path);
    output.write(static_cast<char const*>(data),
        static_cast<std::streamsize>(size));
    require(static_cast<bool>(output), "写入文件失败: " + path);
}

inline std::string dimsToString(nvinfer1::Dims const& dims)
{
    std::ostringstream stream;
    stream << '{';
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        if (i != 0)
        {
            stream << ", ";
        }
        stream << dims.d[i];
    }
    stream << '}';
    return stream.str();
}

inline std::size_t volume(nvinfer1::Dims const& dims)
{
    std::size_t result = 1;
    for (int32_t i = 0; i < dims.nbDims; ++i)
    {
        require(dims.d[i] >= 0,
            "张量仍包含未指定的动态维度: " + dimsToString(dims));
        result *= static_cast<std::size_t>(dims.d[i]);
    }
    return result;
}

inline std::size_t dataTypeSize(nvinfer1::DataType type)
{
    using nvinfer1::DataType;
    switch (type)
    {
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT8: return 1;
    case DataType::kINT32: return 4;
    case DataType::kBOOL: return 1;
    case DataType::kUINT8: return 1;
    case DataType::kFP8: return 1;
    case DataType::kBF16: return 2;
    case DataType::kINT64: return 8;
    case DataType::kINT4: return 1;
    case DataType::kFP4: return 1;
    case DataType::kE8M0: return 1;
    }
    throw std::runtime_error("不支持的 TensorRT 数据类型");
}

inline TrtUniquePtr<nvinfer1::ICudaEngine> loadEngine(
    std::string const& path, Logger& logger)
{
    auto bytes = readBinary(path);
    TrtUniquePtr<nvinfer1::IRuntime> runtime{
        nvinfer1::createInferRuntime(logger)};
    require(static_cast<bool>(runtime), "创建 TensorRT runtime 失败");
    TrtUniquePtr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(bytes.data(), bytes.size())};
    require(static_cast<bool>(engine), "反序列化 engine 失败: " + path);
    return engine;
}

struct DeviceBuffer
{
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t byteCount)
        : bytes(byteCount)
    {
        checkCuda(cudaMalloc(&data, bytes), "cudaMalloc");
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data(other.data), bytes(other.bytes)
    {
        other.data = nullptr;
        other.bytes = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            if (data)
            {
                cudaFree(data);
            }
            data = other.data;
            bytes = other.bytes;
            other.data = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        if (data)
        {
            cudaFree(data);
        }
    }

    void* data{nullptr};
    std::size_t bytes{0};
};

inline void fillInput(DeviceBuffer const& buffer, nvinfer1::DataType type)
{
    std::size_t const count = buffer.bytes / dataTypeSize(type);
    if (type == nvinfer1::DataType::kFLOAT)
    {
        std::vector<float> host(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            host[i] = static_cast<float>(i % 97) / 97.0F;
        }
        checkCuda(cudaMemcpy(buffer.data, host.data(), buffer.bytes,
                      cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D float)");
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        std::vector<__half> host(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            host[i] = __float2half(static_cast<float>(i % 97) / 97.0F);
        }
        checkCuda(cudaMemcpy(buffer.data, host.data(), buffer.bytes,
                      cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D half)");
    }
    else
    {
        checkCuda(cudaMemset(buffer.data, 0, buffer.bytes), "cudaMemset");
    }
}

inline void printValues(DeviceBuffer const& buffer, nvinfer1::DataType type,
    std::size_t maxCount = 10)
{
    std::size_t const count
        = std::min(maxCount, buffer.bytes / dataTypeSize(type));
    if (type == nvinfer1::DataType::kFLOAT)
    {
        std::vector<float> host(count);
        checkCuda(cudaMemcpy(host.data(), buffer.data, count * sizeof(float),
                      cudaMemcpyDeviceToHost),
            "cudaMemcpy(D2H float)");
        for (float value : host)
        {
            std::cout << ' ' << std::fixed << std::setprecision(6) << value;
        }
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        std::vector<__half> host(count);
        checkCuda(cudaMemcpy(host.data(), buffer.data, count * sizeof(__half),
                      cudaMemcpyDeviceToHost),
            "cudaMemcpy(D2H half)");
        for (__half value : host)
        {
            std::cout << ' ' << std::fixed << std::setprecision(6)
                      << __half2float(value);
        }
    }
    else
    {
        std::cout << " <仅展示 FLOAT/HALF 输出>";
    }
    std::cout << '\n';
}

} // namespace trt_demo

