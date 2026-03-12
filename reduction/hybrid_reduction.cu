// Compile with: hipcc reduction/hybrid_reduction.cu -o hybrid_reduction

#include <stdio.h>
#include <hip/hip_runtime.h>

#define THREAD_PER_BLOCK 256
#define WARP_SIZE 32

#define FULL_MASK 0xFFFFFFFFFFFFFFFF
// #define FULL_MASK 0xFFFFFFFF

__global__ void warp_reduce(float* output, const float* input, int N) {
    extern __shared__ float buffer[];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // copy data to shared memory
    if(global_id < N) {
        buffer[local_id] = input[global_id];
    } else {
        buffer[local_id] = 0.0f;
    }
    __syncthreads();

    // halving stride reduction until we have only 32 threads left
    for(int stride = THREAD_PER_BLOCK / 2; stride >= WARP_SIZE; stride >>= 1) {
        if(local_id < stride) {
            buffer[local_id] += buffer[local_id + stride];
        }
        __syncthreads();
    }

    if(local_id < WARP_SIZE) {
        // each thread in the warp places its value in a register
        float v = buffer[local_id];
        // tree reduction over the warp
        for(int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
            v += __shfl_down_sync(FULL_MASK, v, stride);
        }

        // local_id == 0 on each block all collapse into global_id == 0
        if (local_id == 0) {
            atomicAdd(output, v);
        }
    }
}

int main() {
    constexpr int N = 1024;
    float h_input[N];

    // initialize array to something
    for(int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // allocate device memory
    float *d_output;
    float *d_input;
    hipMalloc(&d_output, sizeof(float));
    hipMalloc(&d_input, N * sizeof(float));

    // copy data to device
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);

    // compute block and grid size
    int number_of_blocks = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    
    // execute kernel
    warp_reduce<<<number_of_blocks, THREAD_PER_BLOCK, THREAD_PER_BLOCK * sizeof(float)>>>(d_output, d_input, N);

    // copy output back
    float result;
    hipMemcpy(&result, d_output, sizeof(float), hipMemcpyDeviceToHost);
    printf("result is %f, expected is %f, diff is %f\n", result, float(N), result - float(N));

    //release memory
    hipFree(d_output);
    hipFree(d_input);
    return 0;    
}