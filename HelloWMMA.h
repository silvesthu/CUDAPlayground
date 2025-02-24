#include "Common.h"

namespace HelloWMMA
{
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    constexpr int A_count = M * K;
    constexpr int B_count = K * N;
    constexpr int C_count = M * N;
    constexpr int D_count = M * N;

    // -------------------- Traditional CUDA Kernel --------------------
    __global__ void traditional_nn(half* a, half* b, float* c, float* d)
    {
        int row = threadIdx.y;
        int col = threadIdx.x;

        float sum = 0.0f;
        for (int i = 0; i < K; i++)
            sum = sum + float(a[row * K + i] * b[i * N + col]);
        d[row * N + col] = sum + c[row * N + col];              // AxB+C
    }

    // -------------------- Cooperative Matrix Kernel (Tensor Cores) --------------------
    __global__ void coop_matrix_nn(half* a, half* b, float* c, float* d)
    {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> A;
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> B;
        wmma::fragment<wmma::accumulator, M, N, K, float> C;
        wmma::fragment<wmma::accumulator, M, N, K, float> D;

        wmma::load_matrix_sync(A, a, M);
        wmma::load_matrix_sync(B, b, K);
        wmma::load_matrix_sync(C, c, M, wmma::mem_row_major);

        wmma::mma_sync(D, A, B, C);                             // AxB+C

        wmma::store_matrix_sync(d, D, N, wmma::mem_row_major);
    }

    void Run()
    {
        printf("** HelloWMMA **\n\n");

        size_t A_bytes = A_count * sizeof(half);
        size_t B_bytes = B_count * sizeof(half);
        size_t C_bytes = C_count * sizeof(float);
        size_t D_bytes = D_count * sizeof(float);

        half* A_host = (half*)malloc(A_bytes);
        half* B_host = (half*)malloc(B_bytes);
        float* C_host = (float*)malloc(C_bytes);
        float* D_host = (float*)malloc(D_bytes);

        for (int i = 0; i < A_count; i++) A_host[i] = i * 0.1f;
        for (int i = 0; i < B_count; i++) B_host[i] = i * 0.01f;
        for (int i = 0; i < C_count; i++) C_host[i] = i * 0.03f;

        printf("A = \n");
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < K; c++)
                printf("%.2f ", (float)A_host[c + r * K]);
            printf("\n");
        }
        printf("\n");
        printf("B = \n");
        for (int r = 0; r < K; r++)
        {
            for (int c = 0; c < N; c++)
                printf("%.2f ", (float)B_host[c + r * N]);
            printf("\n");
        }
        printf("\n");
        printf("C = \n");
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
                printf("%.2f ", C_host[c + r * N]);
            printf("\n");
        }
        printf("\n");

        half* A_device;
        half* B_device;
        float* C_device;
        float* D_device;
        CUDA_CHECK(cudaMalloc(&A_device, A_bytes));
        CUDA_CHECK(cudaMalloc(&B_device, B_bytes));
        CUDA_CHECK(cudaMalloc(&C_device, C_bytes));
        CUDA_CHECK(cudaMalloc(&D_device, D_bytes));

        CUDA_CHECK(cudaMemcpy(A_device, A_host, A_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B_device, B_host, B_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(C_device, C_host, C_bytes, cudaMemcpyHostToDevice));

        // -------- Launch Traditional Kernel --------
        dim3 grid_dim(1);
        dim3 block_dim(N, M);
        traditional_nn << <grid_dim, block_dim >> > (A_device, B_device, C_device, D_device);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(D_host, D_device, D_bytes, cudaMemcpyDeviceToHost));
        printf("Traditional Kernel Output = \n");
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
                printf("%.2f ", D_host[c + r * N]);
            printf("\n");
        }

        printf("\n");

        // -------- Launch Cooperative Matrix Kernel --------
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        dim3 coop_grid_dim(1);
        dim3 coop_block_dim(prop.warpSize);         // Should be multiple of wrapSize = 32 
        coop_matrix_nn << <coop_grid_dim, coop_block_dim >> > (A_device, B_device, C_device, D_device);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(D_host, D_device, D_bytes, cudaMemcpyDeviceToHost));
        printf("Coop-Matrix Kernel Output = \n");
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < N; c++)
                printf("%.2f ", D_host[c + r * N]);
            printf("\n");
        }

        // Cleanup
        cudaFree(A_device); cudaFree(B_device); cudaFree(C_device); cudaFree(D_device);
        free(A_host); free(B_host); free(C_host); free(D_host);

        printf("\n");
    }
}