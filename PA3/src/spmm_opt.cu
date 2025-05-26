#include "spmm_opt.h"

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin,
                                float *vout, int num_v, int INFEATURE) {
  __shared__ int shared_idx[WARP_SIZE];
  __shared__ float shared_val[WARP_SIZE];
  int row = blockIdx.x;
  int col = blockIdx.y * WARP_SIZE + threadIdx.y;
  if (row >= num_v)
    return;

  int start = ptr[row];
  int end = ptr[row + 1];

  float sum = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.y < end) {
      shared_idx[threadIdx.y] = idx[i + threadIdx.y];
      shared_val[threadIdx.y] = val[i + threadIdx.y];
    }
    __syncthreads();
    for (int j = 0; j < min(WARP_SIZE, end - i); j++) {
      sum += shared_val[j] * vin[shared_idx[j] * INFEATURE + col];
    }
  }
  vout[row * INFEATURE + col] = sum;
}

__global__ void spmm_kernel_opt_2x(int *ptr, int *idx, float *val, float *vin,
                                   float *vout, int num_v, int INFEATURE) {
  __shared__ int shared_idx[WARP_SIZE];
  __shared__ float shared_val[WARP_SIZE];
  int row = blockIdx.x;
  int col = blockIdx.y * WARP_SIZE * 2 + threadIdx.y;
  if (row >= num_v)
    return;

  int start = ptr[row];
  int end = ptr[row + 1];

  float sum0 = 0;
  float sum1 = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.y < end) {
      shared_idx[threadIdx.y] = idx[i + threadIdx.y];
      shared_val[threadIdx.y] = val[i + threadIdx.y];
    }
    __syncthreads();
    for (int j = 0; j < min(WARP_SIZE, end - i); j++) {
      sum0 += shared_val[j] * vin[shared_idx[j] * INFEATURE + col];
      sum1 += shared_val[j] * vin[shared_idx[j] * INFEATURE + col + WARP_SIZE];
    }
  }
  vout[row * INFEATURE + col] = sum0;
  vout[row * INFEATURE + col + WARP_SIZE] = sum1;
}

__global__ void spmm_kernel_opt_s(int *ptr, int *idx, float *val, float *vin,
                                  float *vout, int num_v, int INFEATURE,
                                  int *rows, int *starts, int *ends) {
  __shared__ int shared_idx[WARP_SIZE];
  __shared__ float shared_val[WARP_SIZE];

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= num_v)
    return;

  int row = rows[id];
  int col = blockIdx.y * WARP_SIZE + threadIdx.y;
  int start = starts[id];
  int end = ends[id];

  float sum = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.y < end) {
      shared_idx[threadIdx.y] = idx[i + threadIdx.y];
      shared_val[threadIdx.y] = val[i + threadIdx.y];
    }
    __syncthreads();
    for (int j = 0; j < min(WARP_SIZE, end - i); j++) {
      sum += shared_val[j] * vin[shared_idx[j] * INFEATURE + col];
    }
  }
  atomicAdd(&vout[row * INFEATURE + col], sum);
}

__global__ void spmm_kernel_opt_s_2x(int *ptr, int *idx, float *val, float *vin,
                                     float *vout, int num_v, int INFEATURE,
                                     int *rows, int *starts, int *ends) {
  __shared__ int shared_idx[WARP_SIZE];
  __shared__ float shared_val[WARP_SIZE];

  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= num_v)
    return;

  int row = rows[id];
  int col = blockIdx.y * WARP_SIZE * 2 + threadIdx.y;
  int start = starts[id];
  int end = ends[id];

  float sum0 = 0;
  float sum1 = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.y < end) {
      shared_idx[threadIdx.y] = idx[i + threadIdx.y];
      shared_val[threadIdx.y] = val[i + threadIdx.y];
    }
    __syncthreads();
    for (int j = 0; j < min(WARP_SIZE, end - i); j++) {
      sum0 += shared_val[j] * vin[shared_idx[j] * INFEATURE + col];
      sum1 += shared_val[j] * vin[shared_idx[j] * INFEATURE + col + WARP_SIZE];
    }
  }
  atomicAdd(&vout[row * INFEATURE + col], sum0);
  atomicAdd(&vout[row * INFEATURE + col + WARP_SIZE], sum1);
}

void SpMMOpt::preprocess(float *vin, float *vout) {
  if (feat_in == 32) {
    speedup = 1;
    if (num_v == NUMV_COLLAB || num_v == NUMV_CITATION ||
        num_v == NUMV_PRODUCTS || num_v == NUMV_WIKIKG2) {
      slice = false;
    } else {
      slice = true;
    }
  } else if (feat_in == 256) {
    switch (num_v) {
    case NUMV_PROTEIN:
      speedup = 1;
      slice = false;
      break;
    case NUMV_REDDIT_DGL:
    case NUMV_AMAZON_COGDL:
      speedup = 1;
      slice = true;
      break;
    case NUMV_COLLAB:
    case NUMV_CITATION:
    case NUMV_PPA:
    case NUMV_PRODUCTS:
    case NUMV_YOUTUBE:
    case NUMV_YELP:
    case NUMV_WIKIKG2:
      speedup = 2;
      slice = false;
      break;
    case NUMV_ARXIV:
    case NUMV_DDI:
    case NUMV_AM:
      speedup = 2;
      slice = true;
      break;
    default:
      speedup = 2;
      slice = false;
      break;
    }
  } else {
    speedup = 1;
    slice = false;
  }
  block.x = 1;
  block.y = WARP_SIZE;
  grid.y = (feat_in + WARP_SIZE * speedup - 1) / (WARP_SIZE * speedup);
  if (!slice) {
    grid.x = num_v;
  } else {
    int *h_ptr = new int[num_v + 1];
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost));
    num_s = 0;
    for (int i = 0; i < num_v; i++) {
      num_s += (h_ptr[i + 1] - h_ptr[i] + SLICE_SIZE - 1) / SLICE_SIZE;
    }
    int *h_rows = new int[num_s];
    int *h_starts = new int[num_s];
    int *h_ends = new int[num_s];
    int idx = 0;
    for (int i = 0; i < num_v; i++) {
      int start = h_ptr[i];
      int end = h_ptr[i + 1];
      for (int j = start; j < end; j += SLICE_SIZE) {
        h_rows[idx] = i;
        h_starts[idx] = j;
        h_ends[idx] = min(end, j + SLICE_SIZE);
        idx++;
      }
    }
    checkCudaErrors(cudaMalloc(&d_rows, num_s * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_starts, num_s * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_ends, num_s * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_rows, h_rows, num_s * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_starts, h_starts, num_s * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ends, h_ends, num_s * sizeof(int),
                               cudaMemcpyHostToDevice));
    grid.x = num_s;
    delete[] h_ptr;
    delete[] h_rows;
    delete[] h_starts;
    delete[] h_ends;
  }
}

void SpMMOpt::run(float *vin, float *vout) {
  if (speedup == 1 && !slice) {
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                     feat_in);
  } else if (speedup == 2 && !slice) {
    spmm_kernel_opt_2x<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                        feat_in);
  } else if (speedup == 1 && slice) {
    spmm_kernel_opt_s<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_s,
                                       feat_in, d_rows, d_starts, d_ends);
  } else if (speedup == 2 && slice) {
    spmm_kernel_opt_s_2x<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_s,
                                          feat_in, d_rows, d_starts, d_ends);
  } else {
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                     feat_in);
  }
}