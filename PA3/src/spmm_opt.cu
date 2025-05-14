#include "spmm_opt.h"

#define WARP_SIZE 32

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin,
                                float *vout, int num_v, int INFEATURE) {
  __shared__ int shared_idx[WARP_SIZE];
  __shared__ float shared_val[WARP_SIZE];
  int row = blockIdx.x;
  int col = blockIdx.y * WARP_SIZE + threadIdx.y;
  if (row >= num_v)
    return;

  int row_start = ptr[row];
  int row_end = ptr[row + 1];

  float sum = 0;

  for (int i = row_start; i < row_end; i += WARP_SIZE) {
    int p = i + threadIdx.y;
    if (p < row_end) {
      shared_idx[threadIdx.y] = idx[p];
      shared_val[threadIdx.y] = val[p];
    }
    __syncthreads();
    for (int j = 0; j < min(WARP_SIZE, row_end - i); j++) {
      sum += shared_val[j] * vin[shared_idx[j] * INFEATURE + col];
    }
  }
  vout[row * INFEATURE + col] = sum;
}

void SpMMOpt::preprocess(float *vin, float *vout) {
  block.x = 1;
  block.y = WARP_SIZE;
  grid.x = num_v;
  grid.y = (feat_in - 1) / WARP_SIZE + 1;
}

void SpMMOpt::run(float *vin, float *vout) {
  spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                   feat_in);
}