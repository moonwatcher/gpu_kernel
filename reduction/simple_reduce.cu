// Compile with: hipcc reduction/simple_reduce.cu -o simple_reduce

#include <stdio.h>
#include <hip/hip_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void simple_reduce(float* output, const float* input, int N) {
    extern __shared__ float buffer[];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // put input in local memory
    if(global_id < N) {
        buffer[local_id] = input[global_id];
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
    float h_input[N];

    // compute block and grid size
    int number_of_blocks = (N + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;

    // initialize arrays to something
    for(int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // allocate device memory
    float *d_output;
    float *d_input;
    hipMalloc(&d_output, number_of_blocks * sizeof(float));
    hipMalloc(&d_input, N * sizeof(float));

    // copy data to device
    hipMemcpy(d_input, h_input, N * sizeof(float), hipMemcpyHostToDevice);

    // execute kernel
    simple_reduce<<<number_of_blocks, THREAD_PER_BLOCK, THREAD_PER_BLOCK * sizeof(float)>>>(d_output, d_input, N);

    // copy output back
    float result;
    hipMemcpy(&result, d_output, sizeof(float), hipMemcpyDeviceToHost);
    printf("result is %f, expected is %f, diff is %f\n", result, float(N), result - float(N));

    //release memory
    hipFree(d_output);
    hipFree(d_input);
    return 0;    
}