// Compile with: hipcc matmul/tiled.cu -o tiled
/*
Optimization techniques for GPU matrix multiplication:
Naive: each thread reads full row/column from global memory → O(MK + KN) global reads per block
Tiled: shared memory tiles, TILE×TILE block → TILE× reduction in global reads
Coarsened: each thread computes multiple outputs → bigger effective tile, more reuse, accumulators in registers
Pipelined: double/triple buffer to overlap load and compute → hide memory latency
Vectorized loads + bank conflict avoidance: maximize memory throughput
Tensor Cores: replace scalar FMA loop with hardware matrix instructions → 10-16× throughput jump
*/

#include <stdio.h>
#include <hip/hip_runtime.h>

#define TILE 16

__global__ void tiled_matmul(float* C, const float* A, const float* B, const int M, const int K, const int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // the row and column the thread is responsible for computing in the output matrix C
    // column is mapped to x grid dimension so that threads in the same warp access consecutive memory locations in B, 
    // which is stored in row-major order
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // sum is the value of C[row][col] computed by this thread, stored in a register
    float sum = 0.0f;

    // Iterate over tiles along the K dimension
    // t is the tile index
    // NUMBER_OF_TILES makes sure we have enough tiles to cover K elements
    int NUMBER_OF_TILES = (K + TILE - 1) / TILE;
    for (int t = 0; t < NUMBER_OF_TILES; t++) {

        // beginning of the tile in the K dimension
        int tile_offset = t * TILE;

        // Load a tile from A and B to shared memroy
        // In every tile, one thread loads one element of A and one element of B into shared memory
        // A is MxK, B is KxN, C is MxN
        // Thread (threadIdx.y, threadIdx.x) computes C[row][col]
        // Where row = blockIdx.y * TILE + threadIdx.y
        // And col = blockIdx.x * TILE + threadIdx.x
        if (row < M && tile_offset + threadIdx.x < K)
            // [threadIdx.y][threadIdx.x] is local adressing in the block
            // Load A[row][tile_offset + threadIdx.x] into shared memory
            // For a 16x16 tile we are loading 2 rows of 16 elements, 
            // for each row the only thing that changes across threads in a warp is threadIdx.x.
            // But once we switch to the next row there is gap. So this requires 2 memory transactions to load a full warp.
            // a 32x8 tile will solve this.
            As[threadIdx.y][threadIdx.x] = A[(row * K) + (tile_offset + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (tile_offset + threadIdx.y < K && col < N)
            // [threadIdx.y][threadIdx.x] is local adressing in the block
            // Load B[tile_offset + threadIdx.y][col] into shared memory
            // For a 16x16 tile we are loading 2 columns of 16 elements, 
            // for each column the only thing that changes across threads in a warp is col = blockIdx.x * TILE + threadIdx.x, 
            // so again two memory transactions are required to load a full warp. A 32x8 tile will solve this.
            Bs[threadIdx.y][threadIdx.x] = B[((tile_offset + threadIdx.y) * N) + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // Barrier 1: wait for all thread to finish loading to shared memory
        __syncthreads();
 
        // Accumulate partial dot product from shared memory for this tile
        for (int k = 0; k < TILE; k++) {
            // reading from As can potentially cause bank conflicts, 
            // but since all threads in a warp read the same k value, they will all access the same column of As, so there will be no bank conflicts.
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Barrier 2: wait for all thread to finish accumulation before loading the next tile to shared memory
        __syncthreads();
    }

    // Write sum to C[row][col]
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
};

int main() {
    constexpr int M = 1000;
    constexpr int N = 1200;
    constexpr int K = 350;

    printf("Running tiled matrix multiplication with M=%d, N=%d, K=%d\n", M, N, K);
    printf("A total size is %d.\n", M*K);
    printf("B total size is %d.\n", K*N);
    printf("C total size is %d.\n", M*N);

    float h_A[M * K];
    float h_B[K * N];
    float h_C[M * N];

    // initialize arrays to something
    for(int i = 0; i < M * K; i++) {
        h_A[i] = 1.0f;
    }
    for(int i = 0; i < K * N; i++) {
        h_B[i] = 1.0f;
    }
    for(int i = 0; i < M * N; i++) {
        h_C[i] = 0.0f;
    }

    // allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));

    // copy data to device
    hipMemcpy(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice);
    
   // Launch: each block computes a TILE×TILE output tile
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    tiled_matmul<<<grid, block>>>(d_C, d_A, d_B, M, K, N);

    // copy output back
    hipMemcpy(h_C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    //print some output to verify correctness
    int errors = 0;
    for (int i = 0; i < M * N; i++) {
        if (h_C[i] != K) {
            printf("Error at C[%d] = %f, expected %f\n", i, h_C[i], (float)K);
            errors++;
        }
    }
    if (errors == 0) {
        printf("All values are correct!\n");
    } else {
        printf("Total errors: %d\n", errors);
    }

    //release memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);

    return 0;    
};