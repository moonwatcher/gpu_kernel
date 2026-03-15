#include <stdio.h>
#include <hip/hip_runtime.h>

#define TILE 4

__global__ void free_form_valid_convolution_2d (
    float* output,
    const float* input, 
    const float* filter, 
    int width, 
    int height,
    int filter_width,
    int filter_height) {
    extern __shared__ float buffer[];

    const int width_halo = filter_width / 2;
    const int height_halo = filter_height / 2;
    const int buffer_width = TILE + 2 * width_halo;
    const int buffer_height = TILE + 2 * height_halo;
    const int out_width = width - 2 * width_halo;
    const int out_height = height - 2 * height_halo; 

    // Load tile into shared memeory buffer
    for(int buffer_row = threadIdx.y; buffer_row < buffer_height; buffer_row += TILE) {
        for(int buffer_col = threadIdx.x; buffer_col < buffer_width; buffer_col += TILE) {
            int in_row = blockIdx.y * blockDim.y + buffer_row;
            int in_col = blockIdx.x * blockDim.x + buffer_col;
            buffer[buffer_row * buffer_width + buffer_col] = input[in_row * width + in_col];
        }
    }
    __syncthreads();

    // compute convolution
    float sum = 0.0f;
    for(int row = 0; row < filter_height; row++) {
        for(int col = 0; col < filter_width; col++) {
            sum += filter[row * filter_width + col] * buffer[(threadIdx.y + row) * buffer_width + (threadIdx.x + col)];
        }
    }

    // write to output
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    if(out_row < out_height && out_col < out_width) {
        output[out_row * out_width + out_col] = sum;
    }
};

int main() {
    constexpr int in_width = 17;
    constexpr int in_height = 13;
    constexpr int filter_width = 3;
    constexpr int filter_height = 5;
    constexpr int width_halo = filter_width / 2;
    constexpr int height_halo = filter_height / 2;
    constexpr int out_width = in_width - 2 * width_halo;
    constexpr int out_height = in_height - 2 * height_halo;
    constexpr int buffer_size = (TILE + 2 * width_halo) * (TILE + 2 * height_halo);

    // host arrays
    float h_input[in_width * in_height];
    float h_output[out_width * out_height];
    float h_filter[filter_width * filter_height];

    // initialize filter
    for(int row = 0; row < filter_height; row++) {
        for(int col = 0; col < filter_width; col++) {
            h_filter[row * filter_width + col] = 1.0f / (filter_width * filter_height);
        }
    }
    
    // initialize input
    for(int row = 0; row < in_height; row++) {
        for(int col = 0; col < in_width; col++) {
            h_input[row * in_width + col] = 1.0f;
        }
    }

    // device arrays
    float* d_input;
    float* d_output;
    float* d_filter;

    hipMalloc(&d_input, in_width * in_height * sizeof(float));
    hipMalloc(&d_output, out_width * out_height * sizeof(float));
    hipMalloc(&d_filter, filter_width * filter_height * sizeof(float));

    hipMemcpy(d_input, h_input, in_width * in_height * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_filter, h_filter, filter_width * filter_height * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((out_width + TILE - 1) / TILE, (out_height + TILE - 1) / TILE);
    free_form_valid_convolution_2d<<<grid, block, buffer_size * sizeof(float)>>>(
        d_output, 
        d_input, 
        d_filter, 
        in_width, 
        in_height, 
        filter_width, 
        filter_height
    );

    hipMemcpy(h_output, d_output, out_width * out_height * sizeof(float), hipMemcpyDeviceToHost);

    //Valid convolution with all 1s input and average filter should give 1.0 for all outputs
    bool valid = true;
    for(int row = 0; row < out_height; row++) {
        for(int col = 0; col < out_width; col++) {
            if (h_output[row * out_width + col] != 1.0f) {
                valid = false;
                printf("Error at output (%d, %d): expected 1.0, got %f\n", row, col, h_output[row * out_width + col]);
            }
        }
    }
    if (valid) {
        printf("All outputs are correct!\n");
    }

    for(int row = 0; row < out_height; row++) {
        for(int col = 0; col < out_width; col++) {
            printf("%f ", h_output[row * out_width + col]);
        }
        printf("\n");
    }

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_filter);
    return 0;
};