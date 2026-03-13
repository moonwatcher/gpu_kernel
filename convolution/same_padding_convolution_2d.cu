/*
Same padding 2D convolution
input is zero padded so the output is the same size as the input. 

Compile: hipcc convolution/same_padding_convolution_2d.cu -o same_padding_convolution_2d
Run: ./same_padding_convolution_2d
*/

#include <stdio.h>
#include <hip/hip_runtime.h>

#define TILE 4
#define HALO 2

__global__ void same_pading_convolution_2d(float* output, const float* input, const float* filter, int width, int height) {
    // The buffer is (blockDim.x + 2 * HALO)x(blockDim.y + 2 * HALO)
    extern __shared__ float buffer[];
    const int filter_size = 2 * HALO + 1;
    const int buffer_width = blockDim.x + 2 * HALO;

    for (int dy = threadIdx.y; dy < blockDim.y + 2 * HALO; dy += TILE) {
        for (int dx = threadIdx.x; dx < blockDim.x + 2 * HALO; dx += TILE) {
            int in_row = blockIdx.y * blockDim.y + dy - HALO;
            int in_col = blockIdx.x * blockDim.x + dx - HALO;
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
    if(out_row < height && out_col < width) {
        output[out_row * width + out_col] = sum;
    }
};

int main() {
    constexpr int width = 11;
    constexpr int height = 13;
    constexpr int filter_size = 2 * HALO + 1;
    constexpr int shared_size = (TILE + 2 * HALO) * (TILE + 2 * HALO);

    float h_input[width * height];
    float h_output[width * height];
    float filter[filter_size * filter_size];

    // initialize input to something
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            h_input[row * width + col] = 1.0f;
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
    hipMalloc(&d_input, width * height * sizeof(float));
    hipMalloc(&d_output, width * height * sizeof(float));
    hipMalloc(&d_filter, filter_size * filter_size * sizeof(float));

    hipMemcpy(d_input, h_input, width * height * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_filter, filter, filter_size * filter_size * sizeof(float), hipMemcpyHostToDevice);

    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    same_pading_convolution_2d<<<grid, block, shared_size * sizeof(float)>>>(
        d_output,
        d_input,
        d_filter,
        width,
        height
    );

    hipMemcpy(h_output, d_output, width * height * sizeof(float), hipMemcpyDeviceToHost);
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            printf("%f ", h_output[row * width + col]);
        }
        printf("\n");
    }
};