/*
valid, no padding 2D convolution
Compute only outputs where the filter fully overlaps the input.
Output size is (width - filter_width + 1) x (height - filter_height + 1).
Buffer size is (tile + filter_width - 1) x (tile + filter_height - 1) to ensure all necessary input values for the tile are loaded.

Compile: hipcc convolution/valid_convolution_2d.cu -o valid_convolution_2d
Run: ./valid_convolution_2d
*/

#include <stdio.h>
#include <hip/hip_runtime.h>

// #define tile 32

template<int tile>
__global__ void valid_convolution_2d (
    float* output,
    const float* input, 
    const float* filter, 
    int width, 
    int height,
    int filter_width,
    int filter_height) {

    extern __shared__ float buffer[];

    /* Buffer size needs to cover all necessary input values for the tile */
    const int buffer_width = tile + filter_width - 1;
    const int buffer_height = tile + filter_height - 1;

    /* Output size for valid convolution is smaller than input */
    const int out_width = width - filter_width + 1;
    const int out_height = height - filter_height + 1; 

    /*  Load tile into shared memory buffer 
        The outer loops are only used by the first few threads to load the extra elements needed for the convolution.
        Both buffer and input indexing is based on the buffer_row and buffer_col, which are based on the thread block indexing.

    */
    for(int buffer_row = threadIdx.y; buffer_row < buffer_height; buffer_row += tile) {
        for(int buffer_col = threadIdx.x; buffer_col < buffer_width; buffer_col += tile) {
            int in_row = blockIdx.y * blockDim.y + buffer_row;
            int in_col = blockIdx.x * blockDim.x + buffer_col;
            buffer[buffer_row * buffer_width + buffer_col] = input[in_row * width + in_col];
        }
    }
    __syncthreads();

    /*  compute convolution sum for the output element corresponding to this thread
        filter indexing is just based on the filter dimensions.
        buffer indexing is the same as the filter indexing, but offset by the thread's position in the tile (threadIdx.x and threadIdx.y)
        to get the correct input values for this output element.
    */
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
    constexpr int tile = 32;
    constexpr int in_width = 800;
    constexpr int in_height = 800;
    constexpr int filter_width = 3;
    constexpr int filter_height = 5;
    constexpr int out_width = in_width - filter_width + 1;
    constexpr int out_height = in_height - filter_height + 1;
    constexpr int buffer_size = (tile + filter_width - 1) * (tile + filter_height - 1);

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

    dim3 block(tile, tile);
    dim3 grid((out_width + tile - 1) / tile, (out_height + tile - 1) / tile);
    valid_convolution_2d<tile><<<grid, block, buffer_size * sizeof(float)>>>(
        d_output, 
        d_input, 
        d_filter, 
        in_width, 
        in_height, 
        filter_width, 
        filter_height
    );

    hipMemcpy(h_output, d_output, out_width * out_height * sizeof(float), hipMemcpyDeviceToHost);

    // valid convolution with all 1s input and average filter should give 1.0 for all outputs
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

    // Print output for debugging
    // for(int row = 0; row < out_height; row++) {
    //     for(int col = 0; col < out_width; col++) {
    //         printf("%f ", h_output[row * out_width + col]);
    //     }
    //     printf("\n");
    // }

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_filter);
    return 0;
};