/*
valid, no padding 2D convolution
Compute only outputs where the filter fully overlaps the input.
Output size is (width - 2 * HALO) x (height - 2 * HALO).

Compile: hipcc convolution/valid_convolution_2d.cu -o valid_convolution_2d
Run: ./valid_convolution_2d
*/

#include <stdio.h>
#include <hip/hip_runtime.h>

#define TILE 4
#define HALO 1

__global__ void valid_convolution_2d(float* output, const float* input, const float* filter, int width, int height) {
    // The buffer is (TILE + 2 * HALO)x(TILE + 2 * HALO)
    extern __shared__ float buffer[];
    const int filter_size = 2 * HALO + 1;
    const int buffer_width = TILE + 2 * HALO;
    const int out_width = width - 2 * HALO;
    const int out_height = height - 2 * HALO;

    for (int dy = threadIdx.y; dy < buffer_width; dy += TILE) {
        for (int dx = threadIdx.x; dx < buffer_width; dx += TILE) {
            int in_row = blockIdx.y * blockDim.y + dy;
            int in_col = blockIdx.x * blockDim.x + dx;
            if(in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
                buffer[dy * buffer_width + dx] = input[in_row * width + in_col];
            } else {
                buffer[dy * buffer_width + dx] = 0.0f;
            }
        }
    }

    __syncthreads();

    // compute convolution
    float sum = 0.0;
    for(int i = 0; i < filter_size; i++) {
        for(int j = 0; j < filter_size; j++) {
            sum += filter[i * filter_size + j] * buffer[(threadIdx.y + i) * buffer_width + (threadIdx.x + j)];
        }
    }
    int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    if(out_row < out_height && out_col < out_width) {
        output[out_row * out_width + out_col] = sum;
    }
};

int main() {
    constexpr int in_width = 13;
    constexpr int in_height = 11;
    constexpr int out_width = in_width - 2 * HALO;
    constexpr int out_height = in_height - 2 * HALO;

    constexpr int filter_size = 2 * HALO + 1;
    constexpr int shared_size = (TILE + 2 * HALO) * (TILE + 2 * HALO);

    float h_input[in_width * in_height];
    float h_output[out_width * out_height];
    float filter[filter_size * filter_size];

    // initialize input to something
    for(int row = 0; row < in_height; row++) {
        for(int col = 0; col < in_width; col++) {
            h_input[row * in_width + col] = 1.0f;
        }
    }

    for(int i = 0; i < filter_size; i++) {
        for(int j = 0; j < filter_size; j++) {
            filter[i * filter_size + j] = 1.0f / float(filter_size * filter_size);
        }
    }

    float* d_input;
    float* d_output;
    float* d_filter;
    hipMalloc(&d_input, in_width* in_height * sizeof(float));
    hipMalloc(&d_output, out_width * out_height * sizeof(float));
    hipMalloc(&d_filter, filter_size * filter_size * sizeof(float));

    hipMemcpy(d_input, h_input, in_width * in_height * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_filter, filter, filter_size * filter_size * sizeof(float), hipMemcpyHostToDevice);

    dim3 grid((out_width + TILE - 1) / TILE, (out_height + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    valid_convolution_2d<<<grid, block, shared_size * sizeof(float)>>>(
        d_output,
        d_input,
        d_filter,
        in_width,
        in_height
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
};