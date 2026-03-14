#include <stdio.h>
#include <hip/hip_runtime.h>

__global__ void convolution_1d(float* output, const float* input, const float* filter, int n, int filter_size) {
    // size of buffer should be blockDim.x + filter_size - 1
    extern __shared__ float buffer[];

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;
    int halo = filter_size / 2;
    int buffer_size = blockDim.x + filter_size - 1;

    // Load left most blockDim.x items in the buffer
    int input_idx = global_id - halo;
    if (input_idx >= 0 && input_idx < n) {
        buffer[local_id] = input[input_idx];
    } else {
        buffer[local_id] = 0.0f;
    }

    // now load the right filter_size - 1 elements
    if (local_id < filter_size - 1) {
        int right_idx = input_idx + blockDim.x;
        if(right_idx >= 0 && right_idx < n) {
            buffer[local_id + blockDim.x] = input[right_idx];
        } else {
            buffer[local_id + blockDim.x] = 0.0f;
        }
    }

    __syncthreads();

    // apply the filter from the buffer
    if(global_id < n) {
        float sum = 0.0;
        for(int j = 0; j < filter_size; j++) {
            sum += filter[j] * buffer[local_id + j];
        }
        output[global_id] = sum;
    }
};

int main() {
    constexpr int thread_per_block = 256;
    constexpr int n = 1024;
    constexpr int filter_size = 5;
    int blocks_per_grid = (n + thread_per_block - 1) / thread_per_block;
    
    float h_input[n];
    float h_output[n];
    float filter[filter_size];

    // initialize input to something
    for(int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }

    for(int i = 0; i < filter_size; i++) {
        filter[i] = 1.0 / float(filter_size);
    }

    float* d_input;
    float* d_output;
    float* d_filter;
    hipMalloc(&d_input, n * sizeof(float));
    hipMalloc(&d_output, n * sizeof(float));
    hipMalloc(&d_filter, filter_size * sizeof(float));

    hipMemcpy(d_input, h_input, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_output, h_output, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_filter, filter, filter_size * sizeof(float), hipMemcpyHostToDevice);

    convolution_1d<<<blocks_per_grid, thread_per_block, (thread_per_block + filter_size - 1) * sizeof(float)>>>(
        d_output,
        d_input,
        d_filter,
        n,
        filter_size
    );
    hipMemcpy(h_output, d_output, n * sizeof(float), hipMemcpyDeviceToHost);
    for(int i = 0; i < n; i++) {
        printf("%f ", h_output[i]);
    }
    printf("\n");

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_filter);
    return 0;
};