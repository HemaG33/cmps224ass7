
#include "common.h"

#include "timer.h"

#define BLOCK_DIM 1024

__global__ void scan_kernel(float* input, float* output, float* partialSums, unsigned int N) {

     __shared__ float temp[2 * BLOCK_DIM];

    unsigned int start = 2 * blockIdx.x * blockDim.x;

    // Load data into shared memory
    if (start + threadIdx.x < N)
        temp[threadIdx.x] = input[start + threadIdx.x];
    else
        temp[threadIdx.x] = 0;
    if (start + blockDim.x + threadIdx.x < N)
        temp[blockDim.x + threadIdx.x] = input[start + blockDim.x + threadIdx.x];
    else
        temp[blockDim.x + threadIdx.x] = 0;

    // Reduction phase
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < 2 * blockDim.x)
            temp[index] += temp[index - stride];
    }

    // Post reduction reverse phase with final adjustment
    for (int stride = BLOCK_DIM / 2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index + stride < 2 * blockDim.x)
            temp[index + stride] += temp[index];
    }
    __syncthreads();

    // Write to output and compute partial sums
    if (start + threadIdx.x < N)
        output[start + threadIdx.x] = (threadIdx.x > 0) ? temp[threadIdx.x - 1] : 0;
    if (start + blockDim.x + threadIdx.x < N)
        output[start + blockDim.x + threadIdx.x] = temp[blockDim.x + threadIdx.x - 1];

    if (partialSums && threadIdx.x == 0)
        partialSums[blockIdx.x] = temp[2 * blockDim.x - 1];
}

__global__ void add_kernel(float* output, float* partialSums, unsigned int N) {

    unsigned int start = 2 * blockDim.x * blockIdx.x;

    // Add partial sum to each element
    if (start + threadIdx.x < N)
        output[start + threadIdx.x] += partialSums[blockIdx.x];
    if (start + blockDim.x + threadIdx.x < N)
        output[start + blockDim.x + threadIdx.x] += partialSums[blockIdx.x];


}

void scan_gpu_d(float* input_d, float* output_d, unsigned int N) {

    Timer timer;

    // Configurations
    const unsigned int numThreadsPerBlock = BLOCK_DIM;
    const unsigned int numElementsPerBlock = 2*numThreadsPerBlock;
    const unsigned int numBlocks = (N + numElementsPerBlock - 1)/numElementsPerBlock;

    // Allocate partial sums
    startTime(&timer);
    float *partialSums_d;
    cudaMalloc((void**) &partialSums_d, numBlocks*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Partial sums allocation time");

    // Call kernel
    startTime(&timer);
    scan_kernel <<< numBlocks, numThreadsPerBlock >>> (input_d, output_d, partialSums_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Scan partial sums then add
    if(numBlocks > 1) {

        // Scan partial sums
        scan_gpu_d(partialSums_d, partialSums_d, numBlocks);

        // Add scanned sums
        add_kernel <<< numBlocks, numThreadsPerBlock >>> (output_d, partialSums_d, N);

    }

    // Free memory
    startTime(&timer);
    cudaFree(partialSums_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}

