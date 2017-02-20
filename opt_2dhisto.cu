#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "opt_2dhisto.h"

// forward declarations
__global__ void opt_2dhisto_kernel(uint32_t *d_data, uint32_t *d_bins);
// end forward declarations

void opt_2dhisto(uint32_t *d_data, uint32_t *d_bins)
{
        static int gridsz = PADDED_INPUT_SIZE / BLOCK_SIZE + 1;
        static dim3 dimgrid(gridsz);
        static dim3 dimblock(BLOCK_SIZE);

        cudaMemset(d_bins, 0, NUM_BINS * sizeof(uint32_t));

        opt_2dhisto_kernel<<<dimgrid, dimblock>>>(d_data, d_bins);
}

/* Include below the implementation of any other functions you need */

__global__ void opt_2dhisto_kernel(uint32_t *d_data, uint32_t *d_bins){
        const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

        const int numThreads = blockDim.x * gridDim.x;

        __shared__ data_tile[32];

        __shared__ local_hist[BLOCK_SIZE];

        // block local loading of shmem

        if(globalTid % PADDED_INPUT_WIDTH < INPUT_WIDTH){
                data_tile[threadIdx.x] = d_data[gobalTid];
        }

        // build local histogram out of shmem
        for(i = 0; i < 32; i++){
                local_hist[data_tile[(threadIdx.x + i) % 32]] += 1; // IMPLICITLY ASSUMES 1 WARP PER BLOCK
        }

        for(int pos = threadIdx.x; pos < NUM_BINS; pos += blockDim.x){
                d_bins[pos] = local_hist[pos];
        }
}

void setup(uint32_t **d_result, uint32_t **d_data, uint32_t **h_data)
{
        int grid_size = (NUM_BINS / BLOCK_SIZE) + 1;
        dim3 dimgrid(grid_size);
        dim3 dimblock(BLOCK_SIZE);

        cudaMalloc((void **) d_result, NUM_BINS * sizeof(uint32_t));
        cudaMalloc((void **) d_data, PADDED_INPUT_SIZE * sizeof(uint32_t));
        // pointers get mutated. Woo double indirection
        for(int i = 0; i < INPUT_HEIGHT; i++){
                cudaMemcpy(*d_data + i * PADDED_INPUT_WIDTH,
                           (*h_data + i * PADDED_INPUT_WIDTH), // ignoring shitty outer array
                           PADDED_INPUT_WIDTH * sizeof(uint32_t),
                           cudaMemcpyHostToDevice);
        }
}

void teardown(uint32_t *d_result, uint8_t *kernel_bins,  uint32_t *d_data)
{
        uint32_t *h_result = (uint32_t *) malloc(NUM_BINS * sizeof(uint32_t));
        cudaMemcpy((void*) h_result,
                   (void*) d_result,
                   NUM_BINS * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        for(int i = 0; i < NUM_BINS; i++){
                kernel_bins[i] = (h_result[i]); //> UINT8_MAX) ? 255 : h_result[i];
        }

        cudaFree(d_data);
        cudaFree(d_result);
}
