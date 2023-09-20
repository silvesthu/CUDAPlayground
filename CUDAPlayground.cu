// CUDAPlayground
// 
// Tutorial
//	Programming Guide https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
//  Warp Functions https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions
//
// Sample
//	CUDA Samples https://github.com/NVIDIA/cuda-samples
//	Optimizing Parallel Reduction in CUDA https://cuvilib.com/Reduction.pdf
//

#include <iostream>

__global__ void cuda_hello() 
{
	// Grid -> Block -> Thread
	int block_size = blockDim.x * blockDim.y * blockDim.z;
	int tid = 
		blockIdx.z * (gridDim.x * gridDim.y) * block_size +
		blockIdx.y * (gridDim.x) * block_size +
		blockIdx.x * block_size +
		threadIdx.z * (blockDim.x * blockDim.y) +
		threadIdx.y * (blockDim.x) + 
		threadIdx.x;
	printf("tid = %2d; blockIdx = %d,%d,%d / %d,%d,%d; threadIdx = %d,%d,%d / %d,%d,%d\n", 
		tid,
		blockIdx.x, blockIdx.y, blockIdx.z, gridDim.x, gridDim.y, gridDim.z,
		threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
}

int main()
{
	dim3 grid_size(1, 1, 1);
	dim3 block_size(8, 4, 1);
	cuda_hello<<<grid_size, block_size>>>();

	return 0;
}
