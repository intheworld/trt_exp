#include <stdio.h>

__global__ void thread_indices() {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    printf("block id [%d], thread id [%d] \n", bid, tid);
    return;
}

__global__ void thread_indices_2d() 
{
    printf("block id [%d, %d, %d], thread id [%d, %d, %d] \n",
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z);
    return;
}


__global__ void thread_indices_3d()
{
    const int tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    const int bid = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;


    int unique_id = bid * blockDim.x * blockDim.y * blockDim.z + tid;

    printf("block id [%d], thread id [%d], unique thread id [%d] \n", bid, tid, unique_id);


    /*
        假设有一栋楼，是用砖块实心堆起来的。那么大楼就是这个grid，一个房间就是一个block，一块砖就是一个thread.
        blockId就是房间号，threadId是砖的编号；gridDim是楼的层数、开间数等；blockDim是房间的长宽高方向上的砖块个数；
    */ 
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    unique_id = z * gridDim.x * blockDim.x * gridDim.y * blockDim.y + y * gridDim.x * blockDim.x + x;
    printf("coordinate in cude (x, y, z): (%d, %d, %d), unique thread id [%d] \n", x, y, z, unique_id);
    return;
}

int main()
{
    thread_indices<<<2, 3>>>();

    dim3 grid_size(2, 3);
    dim3 block_size(3, 4);
    thread_indices_2d<<<grid_size, block_size>>>();

    dim3 grid_size_3d(2, 3, 4);
    dim3 block_size_3d(1, 2, 3);
    thread_indices_3d<<<grid_size_3d, block_size_3d>>>();

    cudaDeviceSynchronize();
    return 0;
}