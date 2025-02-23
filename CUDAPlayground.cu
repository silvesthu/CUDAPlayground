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

#include <cuda_runtime.h>						// Syntax highlight for CUDA keywords
#include <device_launch_parameters.h>			// Syntax highlight for __launch_bounds__, blockDim, gridDim, etc.

#include <mma.h>								// WMMA API (Tensor Cores), also requires Code Generation >= compute_70,sm_70 (Volta)

#include <stdio.h>                              // printf

using namespace nvcuda;

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
        exit(1); \
    } \
} while (0)

__global__ void cuda_hello_world() 
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

////////////////////////////////////////////////////////////////

#define M 16  // Rows in X
#define N 16  // Columns in W (output dim)
#define K 16  // Columns in X / Rows in W

// -------------------- Traditional CUDA Kernel --------------------
__global__ void traditional_nn(half *X, half *W, float *B, float *Y) {
    __shared__ float tileX[M][K];
    __shared__ float tileW[K][N];

    int row = threadIdx.y;
    int col = threadIdx.x;

    float val = 0.0f;
    for (int t = 0; t < (K); t += blockDim.x) {
        tileX[row][col] = X[row * K + t + col];
        tileW[col][t + row] = W[(t + row) * N + col];
        __syncthreads();

        for (int i = 0; i < blockDim.x; i++)
            val += tileX[row][i] * tileW[i][col];
        __syncthreads();
    }
    val += B[col];
    Y[row * N + col] = fmaxf(val, 0.0f);  // ReLU
}

// -------------------- Cooperative Matrix Kernel (Tensor Cores) --------------------
__global__ void coop_matrix_nn(half *X, half *W, float *B, float *Y) {
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, X, K);
    wmma::load_matrix_sync(b_frag, W, N);

    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    for (int i = 0; i < c_frag.num_elements; i++)
        c_frag.x[i] = fmaxf(c_frag.x[i] + B[i % N], 0.0f);  // ReLU + Bias

    wmma::store_matrix_sync(Y, c_frag, N, wmma::mem_row_major);
}

////////////////////////////////////////////////////////////////

int main()
{
    // Hello World
    if (1)
    {
	    dim3 grid_size(1, 1, 1);				// Dispatch in HLSL
	    dim3 block_size(8, 4, 1);				// ThreadGroup in HLSL 
	    cuda_hello_world<<<grid_size, block_size>>>();

        CHECK_CUDA(cudaDeviceSynchronize());    // Wait for execution
        printf("\n");
    }

    ////////////////////////////////////////////////////////////

    // Traditional Kernel vs. Cooperative Matrix Kernel, sample from ChatGPT
    if (1)
    {
        float* X_h, * W_h, * B_h, * Y_h;
        half* X_d, * W_d;
        float* B_d, * Y_d;

        size_t bytes_f = M * K * sizeof(float);
        size_t bytes_out = M * N * sizeof(float);

        X_h = (float*)malloc(bytes_f);
        W_h = (float*)malloc(bytes_f);
        B_h = (float*)malloc(N * sizeof(float));
        Y_h = (float*)malloc(bytes_out);

        for (int i = 0; i < M * K; i++) X_h[i] = 1.0f;
        for (int i = 0; i < K * N; i++) W_h[i] = 1.0f;
        for (int i = 0; i < N; i++) B_h[i] = 1.0f;

        CHECK_CUDA(cudaMalloc(&X_d, bytes_f));
        CHECK_CUDA(cudaMalloc(&W_d, bytes_f));
        CHECK_CUDA(cudaMalloc(&B_d, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&Y_d, bytes_out));

        CHECK_CUDA(cudaMemcpy(X_d, X_h, bytes_f, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(W_d, W_h, bytes_f, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice));

        // -------- Launch Traditional Kernel --------
        dim3 threads(N, M);
        traditional_nn<<<1, threads>>> (X_d, W_d, B_d, Y_d);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(Y_h, Y_d, bytes_out, cudaMemcpyDeviceToHost));
        printf("Traditional Kernel Output (Y[0]): %f\n", Y_h[0]);

        // -------- Launch Cooperative Matrix Kernel --------
        coop_matrix_nn<<<1, 32>>> (reinterpret_cast<half*>(X_d), reinterpret_cast<half*>(W_d), B_d, Y_d);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(Y_h, Y_d, bytes_out, cudaMemcpyDeviceToHost));
        printf("Cooperative Matrix Kernel Output (Y[0]): %f\n", Y_h[0]);

        // Cleanup
        cudaFree(X_d); cudaFree(W_d); cudaFree(B_d); cudaFree(Y_d);
        free(X_h); free(W_h); free(B_h); free(Y_h);
    }

	return 0;
}
