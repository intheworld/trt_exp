cmake_minimum_required(VERSION 3.20)
project(tensorrt_column_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/local/include)
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(TENSORRT_ONNX_LIBRARY nvonnxparser
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers or nvinfer library were not found")
endif()

add_library(trt_demo_common INTERFACE)
target_include_directories(trt_demo_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_demo_common INTERFACE
    ${TENSORRT_LIBRARY}
    CUDA::cudart)
target_compile_options(trt_demo_common INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)

function(add_trt_example target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE trt_demo_common)
endfunction()

add_trt_example(trt_network_api examples/00_network_api/network_api.cpp)

if(TENSORRT_ONNX_LIBRARY)
    add_trt_example(trt_onnx_builder examples/01_onnx_builder/onnx_builder.cpp)
    target_link_libraries(trt_onnx_builder PRIVATE ${TENSORRT_ONNX_LIBRARY})
else()
    message(WARNING "nvonnxparser was not found; trt_onnx_builder will not be built")
endif()

add_trt_example(trt_infer examples/02_infer/infer.cpp)
add_trt_example(trt_precision examples/03_mixed_precision/mixed_precision.cpp)
add_trt_example(trt_dynamic_build examples/04_dynamic_shape/dynamic_build.cpp)
add_trt_example(trt_dynamic_infer examples/04_dynamic_shape/dynamic_infer.cpp)
add_trt_example(trt_refit examples/05_refit/refit.cpp)

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
    add_trt_example(trt_yolo26 examples/08_yolo26/yolo26.cpp)
    target_include_directories(trt_yolo26 PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(trt_yolo26 PRIVATE ${OpenCV_LIBS})
else()
    message(WARNING "OpenCV was not found; trt_yolo26 will not be built")
endif()

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_library(custom_layer_plugin SHARED
        examples/07_custom_plugin/custom_layer_plugin.cu)
    target_include_directories(custom_layer_plugin PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/07_custom_plugin
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(custom_layer_plugin PRIVATE
        ${TENSORRT_LIBRARY}
        CUDA::cudart)
    set_target_properties(custom_layer_plugin PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

    if(TENSORRT_ONNX_LIBRARY)
        add_trt_example(trt_plugin_build examples/07_custom_plugin/build.cpp)
        target_link_libraries(trt_plugin_build PRIVATE
            custom_layer_plugin
            ${TENSORRT_ONNX_LIBRARY})
    endif()

    add_trt_example(trt_plugin_infer examples/07_custom_plugin/infer.cpp)
    target_link_libraries(trt_plugin_infer PRIVATE custom_layer_plugin)

    ## subdirectory
    add_subdirectory(cuda_examples)
else()
    message(WARNING "nvcc not found; custom IPluginV3 examples are disabled")
endif()
