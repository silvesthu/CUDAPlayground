#ifdef __INTELLISENSE__
#define __CUDACC__
#include <cuda_runtime.h>						// Syntax highlight for CUDA keywords
#endif // __INTELLISENSE__

#include <mma.h>								// WMMA API (Tensor Cores), also requires Code Generation >= compute_70,sm_70 (Volta)

#include <stdio.h>                              // printf
#include <assert.h>                             // assert

using namespace nvcuda;

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        exit(1); \
    } \
} while (0)