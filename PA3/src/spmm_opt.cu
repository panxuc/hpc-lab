#include "spmm_opt.h"

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin,
                                float *vout, int num_v, int INFEATURE) {
  int row = blockIdx.x * blockDim.y + threadIdx.y; // one warp per row
  int lane = threadIdx.x;                          // thread in warp
  if (row >= num_v)
    return;

  int row_start = ptr[row];
  int row_end = ptr[row + 1];

  for (int j = lane; j < INFEATURE; j += 32) {
    float sum = 0.0f;
    for (int i = row_start; i < row_end; ++i) {
      int col = idx[i];
      if (col >= 0 && col < num_v)
        sum += val[i] * vin[col * INFEATURE + j];
    }
    // warp-level reduction (optional, since each thread handles different j)
    vout[row * INFEATURE + j] = sum;
  }
}

void SpMMOpt::preprocess(float *vin, float *vout) {
  int WARPS_PER_BLOCK = 4;
  int THREADS_PER_WARP = 32;
  block.x = THREADS_PER_WARP;
  block.y = WARPS_PER_BLOCK;
  grid.x = (num_v + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
}

void SpMMOpt::run(float *vin, float *vout) {
  spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                   feat_in);
}