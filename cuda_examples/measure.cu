#include <stdio.h>
#include "error.cuh"

//#define DOUBLE
#ifdef DOUBLE
typedef double precision;
#else
typedef float precision;
#endif

#define ADD_TIMES (5U)
#define LOOP (100U)

const precision EPSILON = 1.0e-15;
const precision value_a = 1.23;
const precision value_b = 2.34;
const precision value_c = 3.57;

void __global__ add(const precision *x, const precision *y, precision *z, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N) {
        for (int i = 0; i < ADD_TIMES; i++) {
            z[n] = x[n] + y[n];
        }
    }
}


void check(const precision *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n) {
        if (fabs(z[n] - value_c) > EPSILON) {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}


int main(void)
{
    const int N = 100000000;
    const int M = sizeof(precision) * N;

    precision *h_x = (precision *)malloc(M);
    precision *h_y = (precision *)malloc(M);
    precision *h_z = (precision *)malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = value_a;
        h_y[n] = value_b;
    }

    precision *d_x, *d_y, *d_z;

    cudaMalloc((void **)&d_x, M);
    cudaMalloc(&d_y, M);
    cudaMalloc(&d_z, M);

    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = (N - 1) / block_size + 1;

    // 第一次运行时，gpu 与 cpu 处于预热阶段，表现不纳入统计
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    // 利用 cudaEvent 统计时间
    // 基于 cuda event 的计时方式
    // 声明定义 cudaEvent_t 类型变量
    cudaEvent_t start, stop;
    // 调用 cudaEventCreate() 初始化
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    // cudaEventRecord() 记录代表开始的事件
    // 被测试的函数可以是主机函数也可以是核函数
    // CUDA Event 的计时方式会将两者都统计进来
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaEventSynchronize(start));
    for (int i = 0; i < LOOP; ++i)
    {
        add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);
    }

    // 这里测试的是数据传输的时间，
    // 有一些以数据传输为主导的计算，就比如这种计算，传输大数组到设备上，然后只做了一个加法，放到 GPU 上去做就不划算
    // 在进行算法设计的时候一定要将数据传输的时间也考虑进去，尽量减少主机到设备上的传输或者减少设备到主机上的传输，让设备多做计算少进行数据传输 
    // 另外，需要注意的一点是数据传输耗时多与之前说的访存主导并不是同一回事
    // 数据传输发生在主机与设备之间；访存说的是 gpu 的计算核心访问设备内存，访存发生在设备上与主机无关
    // ps：也不能说访存就一定与主机无关，后面还会学统一内存，统一虚拟内存，这两种访存都与主机相关
    // 主机与设备之间的数据传输走 PCIe， 一种利用统一虚拟内存减少访存的优化方法就是让一部分数据走利用率不高的 PCIe

    // CUDA 工具 nvprof，可以对 cuda 程序进行性能剖析
    // nvprof ./bin/measure_performance_data_transfer
    // 如果遇到 Unified Memory profiling failed 之类的错误
    // 可以尝试选项 --unified-memory-profiling off
    // nvprof ./bin/measure_performance_data_transfer --unified-memory-profiling off

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsed_time;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("time_usage = %g ms. \n", elapsed_time);

    CHECK_CUDA(cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost));

    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}