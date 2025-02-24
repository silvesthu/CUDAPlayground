// CUDAPlayground
// 
// Tutorial
//	Programming Guide https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
//  Warp Functions https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions
//  Warp Matrix https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cooperative%2520matrix#warp-matrix-functions
//      Tensor Cores https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
//
// Sample
//	CUDA Samples https://github.com/NVIDIA/cuda-samples
//	Optimizing Parallel Reduction in CUDA https://cuvilib.com/Reduction.pdf
//

#include "HelloCUDA.h"
#include "HelloWMMA.h"
#include "HelloOptix.h"

int main()
{
    HelloCUDA::Run();
	HelloWMMA::Run();
	HelloOptix::Run();

	return 0;
}
