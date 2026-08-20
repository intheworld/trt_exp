add_executable(hello_cuda hello_cuda.cu)

add_executable(add_cuda add_cuda.cu)

add_executable(exec_config exec_config.cu)

add_executable(measure measure.cu)
target_link_libraries(measure PRIVATE trt_demo_common)

add_executable(global_mem global_memory.cu)
target_link_libraries(global_mem PRIVATE trt_demo_common)

add_executable(mem_access mem_access.cu)
target_link_libraries(mem_access PRIVATE trt_demo_common)