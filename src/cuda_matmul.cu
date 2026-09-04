#include "cuda_matmul.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#define TILE_SIZE 16


// Naive CUDA matmul kernel
__global__ void matmulNaiveKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}


// Tiled/shared-memory CUDA matmul kernel
__global__ void matmulTiledKernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N)
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < numTiles; ++tile)
    {
        int aCol = tile * TILE_SIZE + threadIdx.x;
        int bRow = tile * TILE_SIZE + threadIdx.y;

        if (row < M && aCol < K)
        {
            tileA[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];
        }
        else
        {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (bRow < K && col < N)
        {
            tileB[threadIdx.y][threadIdx.x] =
                B[bRow * N + col];
        }
        else
        {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum +=
                tileA[threadIdx.y][k] *
                tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
        C[row * N + col] = sum;
    }
}


void checkCuda(cudaError_t result, const char* message)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(
            std::string(message) + ": " + cudaGetErrorString(result)
        );
    }
}


Tensor cudaMatmul(const Tensor& a, const Tensor& b)
{
    const auto& aShape = a.shape();
    const auto& bShape = b.shape();

    if (aShape.size() != 2 || bShape.size() != 2)
    {
        throw std::runtime_error("cudaMatmul requires 2D tensors");
    }

    int M = static_cast<int>(aShape[0]);
    int K = static_cast<int>(aShape[1]);
    int K2 = static_cast<int>(bShape[0]);
    int N = static_cast<int>(bShape[1]);

    if (K != K2)
    {
        throw std::runtime_error("Matrix dimensions do not match");
    }

    Tensor result({
        static_cast<size_t>(M),
        static_cast<size_t>(N)
    });

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    checkCuda(
        cudaMalloc(&d_A, bytesA),
        "cudaMalloc d_A failed"
    );

    checkCuda(
        cudaMalloc(&d_B, bytesB),
        "cudaMalloc d_B failed"
    );

    checkCuda(
        cudaMalloc(&d_C, bytesC),
        "cudaMalloc d_C failed"
    );

    checkCuda(
        cudaMemcpy(
            d_A,
            a.data(),
            bytesA,
            cudaMemcpyHostToDevice
        ),
        "Copy A to GPU failed"
    );

    checkCuda(
        cudaMemcpy(
            d_B,
            b.data(),
            bytesB,
            cudaMemcpyHostToDevice
        ),
        "Copy B to GPU failed"
    );

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);

    dim3 blocksPerGrid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );


    // Batch-1 / very small row counts perform better with naive kernel.
    // Larger matrices use shared-memory tiling.
    if (M <= 4)
    {
        matmulNaiveKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N
        );
    }
    else
    {
        matmulTiledKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N
        );
    }

    checkCuda(
        cudaGetLastError(),
        "CUDA kernel launch failed"
    );

    checkCuda(
        cudaDeviceSynchronize(),
        "CUDA kernel execution failed"
    );

    checkCuda(
        cudaMemcpy(
            result.data(),
            d_C,
            bytesC,
            cudaMemcpyDeviceToHost
        ),
        "Copy result to CPU failed"
    );

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return result;
}