#ifndef OPT_KERNEL
#define OPT_KERNEL

#define HISTO_WIDTH  1024
#define HISTO_HEIGHT 1
#define NUM_BINS (HISTO_WIDTH * HISTO_HEIGHT)
#define HISTO_LOG 10

#define UINT8_MAX 255


void opt_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[HISTO_HEIGHT*HISTO_WIDTH]);

/* Include below the function headers of any other functions that you implement */

/* kernel is internal to implementation and as such, not exported in header */

#endif
