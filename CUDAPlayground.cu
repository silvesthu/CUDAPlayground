// CUDAPlayground
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

#include <iostream>

__global__ void cuda_hello() 
{
	// Grid -> Block -> Thread
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("blockIdx = %d / %d, threadIdx = %d / %d, tid = %d\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x, tid);
}

int main()
{
	constexpr int grid_size = 2;
	constexpr int block_size = 2;
	cuda_hello<<<grid_size, block_size>>>();

	return 0;
}
