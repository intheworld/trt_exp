#include "trt_demo/common.hpp"

#include <NvInfer.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{

constexpr std::array<char const*, 80> kCocoNames{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"};

struct LetterboxInfo
{
    float scale{};
    int left{};
    int top{};
};

struct Detection
{
    cv::Rect box;
    float score{};
    int classId{};
};

std::vector<float> preprocess(cv::Mat const& image, int inputWidth,
    int inputHeight, LetterboxInfo& info)
{
    info.scale = std::min(static_cast<float>(inputWidth) / image.cols,
        static_cast<float>(inputHeight) / image.rows);
    int const resizedWidth
        = static_cast<int>(std::round(image.cols * info.scale));
    int const resizedHeight
        = static_cast<int>(std::round(image.rows * info.scale));
    info.left = (inputWidth - resizedWidth) / 2;
    info.top = (inputHeight - resizedHeight) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0,
        cv::INTER_LINEAR);
    cv::Mat canvas(inputHeight, inputWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(
        info.left, info.top, resizedWidth, resizedHeight)));
    cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);

    std::vector<float> input(
        static_cast<std::size_t>(3 * inputHeight * inputWidth));
    std::size_t const plane
        = static_cast<std::size_t>(inputHeight * inputWidth);
    for (int y = 0; y < inputHeight; ++y)
    {
        auto const* row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < inputWidth; ++x)
        {
            std::size_t const pixel
                = static_cast<std::size_t>(y * inputWidth + x);
            input[pixel] = static_cast<float>(row[x][0]) / 255.0F;
            input[plane + pixel] = static_cast<float>(row[x][1]) / 255.0F;
            input[2 * plane + pixel]
                = static_cast<float>(row[x][2]) / 255.0F;
        }
    }
    return input;
}

std::vector<Detection> decode(std::vector<float> const& output,
    LetterboxInfo const& letterbox, cv::Size imageSize, float threshold)
{
    trt_demo::require(output.size() % 6 == 0,
        "YOLO26 端到端输出的最后一维应为 6");
    std::vector<Detection> detections;
    for (std::size_t offset = 0; offset < output.size(); offset += 6)
    {
        float const score = output[offset + 4];
        if (score < threshold)
        {
            continue;
        }
        int const classId = static_cast<int>(std::round(output[offset + 5]));
        if (classId < 0 || classId >= static_cast<int>(kCocoNames.size()))
        {
            continue;
        }

        float const x1 = (output[offset] - letterbox.left) / letterbox.scale;
        float const y1 = (output[offset + 1] - letterbox.top) / letterbox.scale;
        float const x2 = (output[offset + 2] - letterbox.left) / letterbox.scale;
        float const y2 = (output[offset + 3] - letterbox.top) / letterbox.scale;
        int const left = std::clamp(
            static_cast<int>(std::round(x1)), 0, imageSize.width - 1);
        int const top = std::clamp(
            static_cast<int>(std::round(y1)), 0, imageSize.height - 1);
        int const right = std::clamp(
            static_cast<int>(std::round(x2)), 0, imageSize.width - 1);
        int const bottom = std::clamp(
            static_cast<int>(std::round(y2)), 0, imageSize.height - 1);
        if (right > left && bottom > top)
        {
            detections.push_back(
                {{left, top, right - left, bottom - top}, score, classId});
        }
    }
    return detections;
}

void drawDetections(cv::Mat& image, std::vector<Detection> const& detections)
{
    for (Detection const& detection : detections)
    {
        cv::rectangle(image, detection.box, cv::Scalar(0, 220, 0), 2);
        std::string const label = std::string(kCocoNames[detection.classId])
            + " " + std::to_string(static_cast<int>(detection.score * 100.0F))
            + "%";
        int baseline{};
        cv::Size const textSize = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        int const labelTop = std::max(detection.box.y, textSize.height + 6);
        cv::rectangle(image,
            cv::Point(detection.box.x, labelTop - textSize.height - 6),
            cv::Point(detection.box.x + textSize.width + 6, labelTop),
            cv::Scalar(0, 220, 0), cv::FILLED);
        cv::putText(image, label,
            cv::Point(detection.box.x + 3, labelTop - 4),
            cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 0), 1,
            cv::LINE_AA);
    }
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr
            << "用法: trt_yolo26 yolo26n.plan input.jpg output.jpg [置信度=0.25]\n";
        return 2;
    }

    try
    {
        float const threshold = argc > 4 ? std::stof(argv[4]) : 0.25F;
        trt_demo::require(threshold >= 0.0F && threshold <= 1.0F,
            "置信度必须位于 [0, 1]");
        cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
        trt_demo::require(!image.empty(), std::string("无法读取图片: ") + argv[2]);

        trt_demo::Logger logger;
        auto engine = trt_demo::loadEngine(argv[1], logger);
        trt_demo::require(engine->getNbIOTensors() == 2,
            "此示例要求 YOLO26 ONNX 恰好包含一个输入和一个输出");
        trt_demo::TrtUniquePtr<nvinfer1::IExecutionContext> context{
            engine->createExecutionContext()};
        trt_demo::require(static_cast<bool>(context), "创建 context 失败");

        char const* inputName{};
        char const* outputName{};
        for (int32_t i = 0; i < engine->getNbIOTensors(); ++i)
        {
            char const* name = engine->getIOTensorName(i);
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                inputName = name;
            }
            else
            {
                outputName = name;
            }
        }
        trt_demo::require(inputName && outputName, "找不到模型输入或输出");
        trt_demo::require(
            engine->getTensorDataType(inputName) == nvinfer1::DataType::kFLOAT
                && engine->getTensorDataType(outputName)
                    == nvinfer1::DataType::kFLOAT,
            "此示例要求 ONNX 输入输出为 FLOAT");

        nvinfer1::Dims const inputDims = context->getTensorShape(inputName);
        nvinfer1::Dims const outputDims = context->getTensorShape(outputName);
        trt_demo::require(inputDims.nbDims == 4 && inputDims.d[0] == 1
                && inputDims.d[1] == 3 && inputDims.d[2] > 0
                && inputDims.d[3] > 0,
            "输入应为固定形状的 NCHW RGB 张量，实际为 "
                + trt_demo::dimsToString(inputDims));
        trt_demo::require(outputDims.nbDims == 3 && outputDims.d[0] == 1
                && outputDims.d[2] == 6,
            "请导出 YOLO26 默认端到端输出 (1, 300, 6)，实际为 "
                + trt_demo::dimsToString(outputDims));

        LetterboxInfo letterbox;
        std::vector<float> input = preprocess(
            image, inputDims.d[3], inputDims.d[2], letterbox);
        std::vector<float> output(trt_demo::volume(outputDims));
        trt_demo::DeviceBuffer deviceInput(input.size() * sizeof(float));
        trt_demo::DeviceBuffer deviceOutput(output.size() * sizeof(float));
        trt_demo::checkCuda(cudaMemcpy(deviceInput.data, input.data(),
                                deviceInput.bytes, cudaMemcpyHostToDevice),
            "cudaMemcpy(H2D YOLO input)");
        trt_demo::require(context->setTensorAddress(inputName, deviceInput.data),
            "绑定输入失败");
        trt_demo::require(context->setTensorAddress(outputName, deviceOutput.data),
            "绑定输出失败");

        cudaStream_t stream{};
        trt_demo::checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        constexpr int kWarmupIterations = 5;
        constexpr int kTimedIterations = 20;
        for (int i = 0; i < kWarmupIterations; ++i)
        {
            trt_demo::require(context->enqueueV3(stream), "预热 enqueueV3 失败");
        }
        trt_demo::checkCuda(cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(warmup)");
        auto const begin = std::chrono::steady_clock::now();
        for (int i = 0; i < kTimedIterations; ++i)
        {
            trt_demo::require(context->enqueueV3(stream), "计时 enqueueV3 失败");
        }
        trt_demo::checkCuda(cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(timing)");
        auto const end = std::chrono::steady_clock::now();
        trt_demo::checkCuda(cudaMemcpy(output.data(), deviceOutput.data,
                                deviceOutput.bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy(D2H YOLO output)");
        trt_demo::checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");

        auto detections
            = decode(output, letterbox, image.size(), threshold);
        drawDetections(image, detections);
        trt_demo::require(cv::imwrite(argv[3], image),
            std::string("无法保存结果图片: ") + argv[3]);
        double const milliseconds
            = std::chrono::duration<double, std::milli>(end - begin).count()
            / kTimedIterations;
        std::cout << "输入 " << inputName << ' '
                  << trt_demo::dimsToString(inputDims) << "，输出 " << outputName
                  << ' ' << trt_demo::dimsToString(outputDims) << '\n';
        std::cout << "检测到 " << detections.size()
                  << " 个目标，预热后 GPU 平均推理 "
                  << std::fixed << std::setprecision(3) << milliseconds
                  << " ms（" << kTimedIterations << " 次）\n已保存 " << argv[3]
                  << '\n';
        for (Detection const& detection : detections)
        {
            std::cout << "  " << kCocoNames[detection.classId] << "  "
                      << std::setprecision(3) << detection.score << "  ["
                      << detection.box.x << ", " << detection.box.y << ", "
                      << detection.box.width << ", " << detection.box.height
                      << "]\n";
        }
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "错误: " << error.what() << '\n';
        return 1;
    }
}
