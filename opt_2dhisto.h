#ifndef OPT_KERNEL
#define OPT_KERNEL

#define HISTO_WIDTH  1024
#define HISTO_HEIGHT 1
#define NUM_BINS (HISTO_WIDTH * HISTO_HEIGHT)
#define HISTO_LOG 10
#define BLOCK_SIZE 32

#define UINT8_MAX 255


/* Include below the function headers of any other functions that you implement */
void opt_2dhisto(uint32_t *d_data, uint8_t *d_bins, int  data_sz);

void setup(uint8_t *d_result, uint32_t *d_data, uint32_t **input);

void teardown(uint8_t *d_result, uint8_t *kernel_bins,  uint32_t *d_data);


/* kernel is internal to implementation and as such, not exported in header */

#endif
