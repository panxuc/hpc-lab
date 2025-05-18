#include "spmm_opt.h"

#define WARP_SIZE 32
#define SLICE_SIZE 256

bool slice = false;
int *d_rows = nullptr, *d_starts = nullptr, *d_ends = nullptr;
int num_slice = 0;

#define NUMV_ARXIV 169343
#define NUMV_COLLAB 235868
#define NUMV_CITATION 2927963
#define NUMV_DDI 4267
#define NUMV_PROTEIN 132534
#define NUMV_PPA 576289
#define NUMV_REDDIT_DGL 232965
#define NUMV_PRODUCTS 2449029
#define NUMV_YOUTUBE 1138499
#define NUMV_AMAZON_COGDL 1569960
#define NUMV_YELP 716847
#define NUMV_WIKIKG2 2500604
#define NUMV_AM 881680

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
    int p = i + threadIdx.y;
    if (p < end) {
      shared_idx[threadIdx.y] = idx[p];
      shared_val[threadIdx.y] = val[p];
    }
    __syncthreads();
    for (int j = 0; j < min(WARP_SIZE, end - i); j++) {
      sum += shared_val[j] * vin[shared_idx[j] * INFEATURE + col];
    }
  }
  vout[row * INFEATURE + col] = sum;
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
    int p = i + threadIdx.y;
    if (p < end) {
      shared_idx[threadIdx.y] = idx[p];
      shared_val[threadIdx.y] = val[p];
    }
    __syncthreads();
    for (int j = 0; j < min(WARP_SIZE, end - i); j++) {
      sum += shared_val[j] * vin[shared_idx[j] * INFEATURE + col];
    }
  }
  atomicAdd(&(vout[row * INFEATURE + col]), sum);
}

void SpMMOpt::preprocess(float *vin, float *vout) {
  block.x = 1;
  block.y = WARP_SIZE;
  grid.y = (feat_in + WARP_SIZE - 1) / WARP_SIZE;
  if (num_v == NUMV_COLLAB || num_v == NUMV_CITATION ||
      num_v == NUMV_PRODUCTS || num_v == NUMV_WIKIKG2) {
    grid.x = num_v;
    slice = false;
  } else {
    int *h_ptr = new int[num_v + 1];
    checkCudaErrors(cudaMemcpy(h_ptr, d_ptr, (num_v + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost));
    num_slice = 0;
    for (int i = 0; i < num_v; i++) {
      num_slice += (h_ptr[i + 1] - h_ptr[i] + SLICE_SIZE - 1) / SLICE_SIZE;
    }
    int *h_rows = new int[num_slice];
    int *h_starts = new int[num_slice];
    int *h_ends = new int[num_slice];
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
    checkCudaErrors(cudaMalloc(&d_rows, num_slice * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_starts, num_slice * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_ends, num_slice * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_rows, h_rows, num_slice * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_starts, h_starts, num_slice * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ends, h_ends, num_slice * sizeof(int),
                               cudaMemcpyHostToDevice));
    grid.x = num_slice;
    delete[] h_ptr;
    delete[] h_rows;
    delete[] h_starts;
    delete[] h_ends;
    slice = true;
  }
}

void SpMMOpt::run(float *vin, float *vout) {
  if (slice) {
    spmm_kernel_opt_s<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout,
                                       num_slice, feat_in, d_rows, d_starts,
                                       d_ends);
  } else {
    spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                     feat_in);
  }
}