#include "spmm_opt.h"
#include <vector>

__global__ void spmm_kernel_opt(int *ptr, int *idx, float *val, float *vin,
                                float *vout, int num_v, int INFEATURE) {
  __shared__ int shared_idx[WARP_SIZE];
  __shared__ float shared_val[WARP_SIZE];
  int row = blockIdx.x;
  int col = blockIdx.y * WARP_SIZE + threadIdx.x;
  int start = ptr[row];
  int end = ptr[row + 1];

  float sum = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.x < end) {
      shared_idx[threadIdx.x] = idx[i + threadIdx.x];
      shared_val[threadIdx.x] = val[i + threadIdx.x];
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
  int col = blockIdx.y * WARP_SIZE * 2 + threadIdx.x;
  int start = ptr[row];
  int end = ptr[row + 1];

  float sum0 = 0;
  float sum1 = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.x < end) {
      shared_idx[threadIdx.x] = idx[i + threadIdx.x];
      shared_val[threadIdx.x] = val[i + threadIdx.x];
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

__global__ void spmm_kernel_opt_p(int *ptr, int *idx, float *val, float *vin,
                                  float *vout, int num_v, int INFEATURE,
                                  int *rows) {
  __shared__ int shared_idx[WARP_SIZE];
  __shared__ float shared_val[WARP_SIZE];
  int id = blockIdx.x;
  int row = rows[id];
  int col = blockIdx.y * WARP_SIZE + threadIdx.x;
  int start = ptr[id];
  int end = ptr[id + 1];

  float sum = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.x < end) {
      shared_idx[threadIdx.x] = idx[i + threadIdx.x];
      shared_val[threadIdx.x] = val[i + threadIdx.x];
    }
    __syncthreads();
    for (int j = 0; j < min(WARP_SIZE, end - i); j++) {
      sum += shared_val[j] * vin[shared_idx[j] * INFEATURE + col];
    }
  }
  vout[row * INFEATURE + col] = sum;
}

__global__ void spmm_kernel_opt_p_2x(int *ptr, int *idx, float *val, float *vin,
                                     float *vout, int num_v, int INFEATURE,
                                     int *rows) {
  __shared__ int shared_idx[WARP_SIZE];
  __shared__ float shared_val[WARP_SIZE];
  int id = blockIdx.x;
  int row = rows[id];
  int col = blockIdx.y * WARP_SIZE * 2 + threadIdx.x;
  int start = ptr[id];
  int end = ptr[id + 1];

  float sum0 = 0;
  float sum1 = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.x < end) {
      shared_idx[threadIdx.x] = idx[i + threadIdx.x];
      shared_val[threadIdx.x] = val[i + threadIdx.x];
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
  int id = blockIdx.x;
  int row = rows[id];
  int col = blockIdx.y * WARP_SIZE + threadIdx.x;
  int start = starts[id];
  int end = ends[id];

  float sum = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.x < end) {
      shared_idx[threadIdx.x] = idx[i + threadIdx.x];
      shared_val[threadIdx.x] = val[i + threadIdx.x];
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
  int id = blockIdx.x;
  int row = rows[id];
  int col = blockIdx.y * WARP_SIZE * 2 + threadIdx.x;
  int start = starts[id];
  int end = ends[id];

  float sum0 = 0;
  float sum1 = 0;

  for (int i = start; i < end; i += WARP_SIZE) {
    if (i + threadIdx.x < end) {
      shared_idx[threadIdx.x] = idx[i + threadIdx.x];
      shared_val[threadIdx.x] = val[i + threadIdx.x];
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
    switch (num_v) {
    case NUMV_WIKIKG2:
      strategy = 0;
      break;
    case NUMV_COLLAB:
    case NUMV_CITATION:
    case NUMV_PPA:
    case NUMV_REDDIT_DGL:
    case NUMV_PRODUCTS:
    case NUMV_AMAZON_COGDL:
    case NUMV_YELP:
      strategy = 1;
      break;
    case NUMV_ARXIV:
    case NUMV_DDI:
    case NUMV_PROTEIN:
    case NUMV_YOUTUBE:
    case NUMV_AM:
      strategy = 2;
      break;
    default:
      strategy = 1;
      break;
    }
  } else if (feat_in == 256) {
    switch (num_v) {
    case NUMV_PROTEIN:
      speedup = 1;
      strategy = 0;
      break;
    case NUMV_COLLAB:
    case NUMV_CITATION:
    case NUMV_YOUTUBE:
    case NUMV_WIKIKG2:
      speedup = 2;
      strategy = 0;
      break;
    case NUMV_PPA:
    case NUMV_REDDIT_DGL:
    case NUMV_PRODUCTS:
    case NUMV_AMAZON_COGDL:
    case NUMV_YELP:
      speedup = 1;
      strategy = 1;
      break;
    case NUMV_ARXIV:
    case NUMV_DDI:
    case NUMV_AM:
      speedup = 2;
      strategy = 2;
      break;
    default:
      speedup = 1;
      strategy = 1;
      break;
    }
  } else {
    speedup = 1;
    strategy = 0;
  }
  block.x = WARP_SIZE;
  grid.y = (feat_in + WARP_SIZE * speedup - 1) / (WARP_SIZE * speedup);
  if (strategy == 0) {
    grid.x = num_v;
  } else if (strategy == 1) {
    std::vector<int> h_ptr(num_v + 1);
    std::vector<int> h_idx(num_e);
    std::vector<float> h_val(num_e);
    std::vector<std::vector<int>> h_col_rows(num_v);
    std::vector<int> h_col_map(num_v, -1);
    std::vector<int> h_row_map(num_v, -1);
    std::vector<int> h_cols(num_v, -1);
    std::vector<int> h_rows(num_v, -1);
    std::vector<int> h_len(num_v, 0);
    std::vector<int> h_ptr_new(num_v + 1);
    std::vector<int> h_idx_new(num_e);
    std::vector<float> h_val_new(num_e);
    checkCudaErrors(cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_idx.data(), d_idx, num_e * sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_val.data(), d_val, num_e * sizeof(float),
                               cudaMemcpyDeviceToHost));
    int new_col = 0;
    int zero_rows = 0;
    for (int i = 0; i < num_v; i++) {
      if (h_ptr[i] == h_ptr[i + 1]) {
        zero_rows++;
        continue;
      }
      for (int j = h_ptr[i]; j < h_ptr[i + 1]; j++) {
        int col = h_idx[j];
        if (h_col_map[col] == -1) {
          h_col_map[col] = new_col++;
          h_cols[h_col_map[col]] = col;
        }
        h_idx[j] = h_col_map[col];
        h_col_rows[h_idx[j]].push_back(i);
      }
    }
    int new_row = 0;
    for (const auto &rows : h_col_rows) {
      for (int i : rows) {
        if (h_row_map[i] == -1) {
          h_row_map[i] = new_row++;
          h_rows[h_row_map[i]] = i;
          h_len[h_row_map[i]] = h_ptr[i + 1] - h_ptr[i];
        }
      }
    }
    h_ptr_new[0] = 0;
    for (int i = 0; i < num_v; i++) {
      h_ptr_new[i + 1] = h_ptr_new[i] + h_len[i];
    }
    for (int i = 0; i < num_v; i++) {
      int row = h_row_map[i];
      if (row == -1)
        continue;
      int k = h_ptr_new[row];
      for (int j = h_ptr[i]; j < h_ptr[i + 1]; j++, k++) {
        h_idx_new[k] = h_cols[h_idx[j]];
        h_val_new[k] = h_val[j];
      }
    }
    checkCudaErrors(cudaMemcpy(d_ptr, h_ptr_new.data(),
                               sizeof(int) * (num_v + 1),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_idx, h_idx_new.data(), sizeof(int) * num_e,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, h_val_new.data(), sizeof(float) * num_e,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc(&d_rows, sizeof(int) * num_v));
    checkCudaErrors(cudaMemcpy(d_rows, h_rows.data(), sizeof(int) * num_v,
                               cudaMemcpyHostToDevice));
    grid.x = num_v - zero_rows;
  } else if (strategy == 2) {
    std::vector<int> h_ptr(num_v + 1);
    checkCudaErrors(cudaMemcpy(h_ptr.data(), d_ptr, (num_v + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost));
    num_s = 0;
    for (int i = 0; i < num_v; i++) {
      num_s += (h_ptr[i + 1] - h_ptr[i] + SLICE_SIZE - 1) / SLICE_SIZE;
    }
    std::vector<int> h_rows(num_s);
    std::vector<int> h_starts(num_s);
    std::vector<int> h_ends(num_s);
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
    checkCudaErrors(cudaMemcpy(d_rows, h_rows.data(), num_s * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_starts, h_starts.data(), num_s * sizeof(int),
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_ends, h_ends.data(), num_s * sizeof(int),
                               cudaMemcpyHostToDevice));
    grid.x = num_s;
  }
}

void SpMMOpt::run(float *vin, float *vout) {
  if (strategy == 0) {
    if (speedup == 1) {
      spmm_kernel_opt<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                       feat_in);
    } else if (speedup == 2) {
      spmm_kernel_opt_2x<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                          feat_in);
    }
  } else if (strategy == 1) {
    if (speedup == 1) {
      spmm_kernel_opt_p<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                         feat_in, d_rows);
    } else if (speedup == 2) {
      spmm_kernel_opt_p_2x<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout,
                                            num_v, feat_in, d_rows);
    }
  } else if (strategy == 2) {
    if (speedup == 1) {
      spmm_kernel_opt_s<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_s,
                                         feat_in, d_rows, d_starts, d_ends);
    } else if (speedup == 2) {
      spmm_kernel_opt_s_2x<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout,
                                            num_s, feat_in, d_rows, d_starts,
                                            d_ends);
    }
  } else {
    spmm_kernel_opt_p<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v,
                                       feat_in, d_rows);
  }
}