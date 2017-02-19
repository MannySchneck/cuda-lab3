#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

void opt_2dhisto( /*define your own function parameters*/ )
{
        /* This function should only contain a call to the GPU
           histogramming kernel. Any memory allocations and
           transfers must be done outside this function */

}

/* Include below the implementation of any other functions you need */

__global__ void opt_2dhisto_kernel(uint8_t *d_result, uint32_t *d_data, size_t data_sz){
        const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

        const int numThreads = blockDim.x * gridDim.x;

        __shared__ unit s_Hist[NUM_BINS];

        // 0 out shared memory for current block
        for (int pos = threadIdx.x;
             pos < NUM_BINS;
             pos += blockDim.x){
                s_Hist[pos] = 0;
        }
        __syncthreads();

        for (int pos = globalTid;
             pos < dataN;
             pos+= numThreads){
                uint32_t data4 = d_data[pos];
                atomicAdd(s_Hist + (data4 >> 0) & 0xFFU, 1);
                atomicAdd(s_Hist + (data4 >> 8) & 0xFFU, 1);
                atomicAdd(s_Hist + (data4 >> 16) & 0xFFU, 1);
                atomicAdd(s_Hist + (data4 >> 24) & 0xFFU, 1);
        }

        // merge partial histograms to global result
        // atomic add will prevent cross-block races
        for(int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x){
                atomicAdd(d_result + pos, s_Hist[pos]);
        }
}
