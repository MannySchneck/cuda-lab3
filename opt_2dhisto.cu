#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "opt_2dhisto.h"

// forward declarations
__global__ void opt_2dhisto_kernel(uint32_t *d_data, uint8_t *d_bins);
// end forward declarations

void opt_2dhisto(uint32_t *d_data, uint8_t *d_bins)
{
        static int gridsz = INPUT_SIZE / BLOCK_SIZE + 1;
        static dim3 dimgrid(gridsz);
        static dim3 dimblock(BLOCK_SIZE);

        cudaMemset(d_bins, 0, NUM_BINS * sizeof(uint8_t));

        opt_2dhisto_kernel<<<dimgrid, dimblock>>>(d_data, d_bins);
}

/* Include below the implementation of any other functions you need */

__global__ void opt_2dhisto_kernel(uint32_t *d_data, uint8_t *d_bins){
        const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

        const int numThreads = blockDim.x * gridDim.x;

        __shared__ uint8_t s_Hist[NUM_BINS];

        // 0 out shared memory for current block
        for (int pos = threadIdx.x;
             pos < NUM_BINS;
             pos += blockDim.x){
                s_Hist[pos] = 0;
        }
        __syncthreads();

        for (int pos = globalTid;
             pos < INPUT_SIZE;
             pos+= numThreads){
                uint32_t data4 = d_data[pos];
                // handle rollover
                atomicAdd(s_Hist + (data4 >> 0) & 0xFFU, 1);
                atomicAdd(s_Hist + (data4 >> 8) & 0xFFU, 1);
                atomicAdd(s_Hist + (data4 >> 16) & 0xFFU, 1);
                atomicAdd(s_Hist + (data4 >> 24) & 0xFFU, 1);
        }

        // merge partial histograms to global result
        // atomic add will prevent cross-block races
        for(int pos = threadIdx.x; pos < NUM_BINS; pos += blockDim.x){
                atomicAdd(d_bins + pos, s_Hist[pos]);
        }
}

void setup(uint8_t *d_result, uint32_t *d_data, uint32_t **input)
{
        int grid_size = (NUM_BINS / BLOCK_SIZE) + 1;
        dim3 dimgrid(grid_size);
        dim3 dimblock(BLOCK_SIZE);

        cudaMalloc((void **) &d_result, NUM_BINS * sizeof(uint8_t));
        cudaMalloc((void **) &d_data, INPUT_SIZE * sizeof(uint32_t));
        cudaMemcpy(d_data, input, INPUT_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void teardown(uint8_t *d_result, uint8_t *kernel_bins,  uint32_t *d_data)
{
cudaMemcpy((void*) kernel_bins,
           (void*) d_result,
           NUM_BINS * sizeof(uint8_t),
           cudaMemcpyDeviceToHost);

cudaFree(d_data);
cudaFree(d_result);
}
