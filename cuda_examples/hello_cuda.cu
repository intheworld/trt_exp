#include <stdio.h>

__global__ void hello_cuda()
{
    printf("hello cuda from GPU.\n");
    return;
}


int main()
{
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}


