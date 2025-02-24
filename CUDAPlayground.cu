//	CUDAPlayground
// 
//	Documents
//		Programming Guide https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
//			Warp Functions https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions
//		Tensor Cores
//			Introduction https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
//			Programming Guide https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cooperative%2520matrix#warp-matrix-functions
//		Optix
//			Programming Guide https://raytracing-docs.nvidia.com/optix9/guide/index.html
//
//	Sample
//		CUDA Samples https://github.com/NVIDIA/cuda-samples
//		Optix Samples https://github.com/nvpro-samples/optix_advanced_samples/blob/master/src/optixIntroduction/README.md
//		Optimization
//			Optimizing Parallel Reduction in CUDA https://cuvilib.com/Reduction.pdf

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
