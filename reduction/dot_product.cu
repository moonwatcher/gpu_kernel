// Compile with: hipcc reduction/dot_product.cu -o dot_product

#include <stdio.h>
#include <hip/hip_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void dot_product(float* output, const float* lhs, const float* rhs, int N) {
    extern __shared__ float buffer[];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // put products in local memory
    if(global_id < N) {
        buffer[local_id] = lhs[global_id] * rhs[global_id];
    } else {
        buffer[local_id] = 0.0f;
    }
    __syncthreads();

    // halving stride reduction
    for(int stride = THREAD_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if(local_id < stride) {
            buffer[local_id] += buffer[local_id + stride];
        }
        __syncthreads();
    }

    // local_id == 0 on each block all collapse into global_id == 0
    if (local_id == 0) {
        atomicAdd(&output[0], buffer[0]);
    }
}

int main() {
    constexpr int N = 1024;
    float h_lhs[N], h_rhs[N];

    // initialize arrays to something
    for(int i = 0; i < N; i++) {
        h_lhs[i] = 1.0f;
        h_rhs[i] = 2.0f;
    }

    // compute block and grid size
    int number_of_blocks = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    printf("N is %d\nThreads per block is %d\nNumber of blocks is %d\n", N, THREAD_PER_BLOCK, number_of_blocks);

    // allocate device memory
    float *d_output;
    float *d_lhs;
    float *d_rhs;
    hipMalloc(&d_output, number_of_blocks * sizeof(float));
    hipMalloc(&d_lhs, N * sizeof(float));
    hipMalloc(&d_rhs, N * sizeof(float));

    // copy data to device
    hipMemcpy(d_lhs, h_lhs, N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_rhs, h_rhs, N * sizeof(float), hipMemcpyHostToDevice);

    // execute kernel
    dot_product<<<number_of_blocks, THREAD_PER_BLOCK, THREAD_PER_BLOCK * sizeof(float)>>>(d_output, d_lhs, d_rhs, N);

    // copy output[0] back, it has the final result after reduction
    float result;
    hipMemcpy(&result, d_output, sizeof(float), hipMemcpyDeviceToHost);
    printf("result is %f, expected is %f, diff is %f\n", result, N * 1.0f * 2.0f, result - N * 1.0f * 2.0f);

    // release memory
    hipFree(d_output);
    hipFree(d_lhs);
    hipFree(d_rhs);
    return 0;    
}