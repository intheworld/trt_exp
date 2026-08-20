#include <stdio.h>
#include <error.cuh>

#define DOUBLE

#ifdef DOUBLE
typedef double real;
#else
typedef float real;
#endif

void __global__ transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}
void __global__ transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
}

void check(const real *array, const int N) 
{
    for (int i = 0; i < N; i++) 
    {
        for (int j = 0; j < N; j++) 
        {
            if (array[i * N + j] != 0.0) 
            {
                printf("Error at (%d, %d): %f\n", i, j, array[i * N + j]);
                return;
            }
        }
    }
    printf("Transpose successful!\n");
}

int main() 
{
    real *a, *b, *host;
    int N = 10000;
    int M = sizeof(real) * N * N;

    host = (real*)malloc(M);
    CHECK_CUDA(cudaMalloc(&a, M));
    CHECK_CUDA(cudaMalloc(&b, M));

    CHECK_CUDA(cudaMemset(a, 0, M));
    CHECK_CUDA(cudaMemset(b, 1, M));

    const dim3 block(32, 32);
    const dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    transpose1<<<grid, block>>>(a, b, N);

    CHECK_CUDA(cudaMemcpy(host, b, M, cudaMemcpyDeviceToHost));

    check(host, N);
    cudaFree(a);
    cudaFree(b);
    free(host);

    return 0;
}