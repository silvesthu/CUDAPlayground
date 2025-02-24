#ifdef __INTELLISENSE__
#define __CUDACC__
#endif // __INTELLISENSE__

#include <cuda_runtime.h>						// Syntax highlight for CUDA keywords

#include <mma.h>								// WMMA API (Tensor Cores), also requires Code Generation >= compute_70,sm_70 (Volta)

#include <stdio.h>                              // printf
#include <assert.h>                             // assert

using namespace nvcuda;

// From optixSDKCurrent.cpp
#define CUDA_CHECK( call )                                                                                             \
    {                                                                                                                  \
        cudaError_t error = call;                                                                                      \
        if( error != cudaSuccess )                                                                                     \
        {                                                                                                              \
            fprintf( stderr, "CUDA call (%s) failed with code %d: %s\n", #call, error, cudaGetErrorString( error ) );  \
            exit( 2 );                                                                                                 \
        }                                                                                                              \
    }