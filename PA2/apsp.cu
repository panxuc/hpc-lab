// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"

#define BLOCK_SIZE 32
#define BATCH_SIZE_PHASE2 6
#define BATCH_SIZE_PHASE3 6
#define MAX_VAL (INT_MAX / 2)

__global__ void phase1(int n, int k, int *graph) {
  __shared__ int shared[BLOCK_SIZE][BLOCK_SIZE];
  int x = threadIdx.x;
  int y = threadIdx.y;
  int i = k * BLOCK_SIZE + y;
  int j = k * BLOCK_SIZE + x;

  shared[y][x] = (i < n && j < n) ? graph[i * n + j] : MAX_VAL;
  __syncthreads();

  if (i < n && j < n) {
    int val = shared[y][x];
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      val = min(val, shared[y][k] + shared[k][x]);
    }
    graph[i * n + j] = val;
  }
}

__global__ void phase2(int n, int k, int *graph) {
  __shared__ int shared_pivot[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int shared_block[BATCH_SIZE_PHASE2][BLOCK_SIZE][BLOCK_SIZE];

  int x = threadIdx.x;
  int y = threadIdx.y;
  int bidx = blockIdx.x;
  int bidy = blockIdx.y;

  int pivot_i = k * BLOCK_SIZE + y;
  int pivot_j = k * BLOCK_SIZE + x;

  shared_pivot[y][x] =
      (pivot_i < n && pivot_j < n) ? graph[pivot_i * n + pivot_j] : MAX_VAL;
  __syncthreads();

  bool is_row = (bidy == 0);
  int block_base_i = is_row ? k : bidx * BATCH_SIZE_PHASE2;
  int block_base_j = is_row ? bidx * BATCH_SIZE_PHASE2 : k;

  for (int p = 0; p < BATCH_SIZE_PHASE2; p++) {
    int block_i = block_base_i + (is_row ? 0 : p);
    int block_j = block_base_j + (is_row ? p : 0);
    // if ((is_row && block_j == k) || (!is_row && block_i == k))
    //   continue;
    int i = block_i * BLOCK_SIZE + y;
    int j = block_j * BLOCK_SIZE + x;
    shared_block[p][y][x] = (i < n && j < n) ? graph[i * n + j] : MAX_VAL;
  }
  __syncthreads();
  for (int p = 0; p < BATCH_SIZE_PHASE2; p++) {
    int block_i = block_base_i + (is_row ? 0 : p);
    int block_j = block_base_j + (is_row ? p : 0);
    // if ((is_row && block_j == k) || (!is_row && block_i == k))
    //   continue;
    int i = block_i * BLOCK_SIZE + y;
    int j = block_j * BLOCK_SIZE + x;
    if (i < n && j < n) {
      int val = shared_block[p][y][x];
      for (int l = 0; l < BLOCK_SIZE; ++l) {
        val = min(val, is_row ? shared_pivot[y][l] + shared_block[p][l][x]
                              : shared_pivot[l][x] + shared_block[p][y][l]);
      }
      graph[i * n + j] = val;
    }
  }
}

__global__ void phase3(int n, int k, int *graph) {
  __shared__ int shared_row[BATCH_SIZE_PHASE3][BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int shared_col[BATCH_SIZE_PHASE3][BLOCK_SIZE][BLOCK_SIZE];

  int x = threadIdx.x;
  int y = threadIdx.y;
  int batch_i = blockIdx.y * BATCH_SIZE_PHASE3;
  int batch_j = blockIdx.x * BATCH_SIZE_PHASE3;
  int kk = k * BLOCK_SIZE;

  for (int p = 0; p < BATCH_SIZE_PHASE3; p++) {
    int row_i = (batch_i + p) * BLOCK_SIZE + y;
    int row_j = kk + x;
    shared_row[p][y][x] =
        (row_i < n && row_j < n) ? graph[row_i * n + row_j] : MAX_VAL;

    int col_i = kk + y;
    int col_j = (batch_j + p) * BLOCK_SIZE + x;
    shared_col[p][y][x] =
        (col_i < n && col_j < n) ? graph[col_i * n + col_j] : MAX_VAL;
  }
  __syncthreads();

  for (int p = 0; p < BATCH_SIZE_PHASE3; p++) {
    for (int q = 0; q < BATCH_SIZE_PHASE3; q++) {
      int i = (batch_i + p) * BLOCK_SIZE + y;
      int j = (batch_j + q) * BLOCK_SIZE + x;

      if (i < n && j < n) {
        int val = graph[i * n + j];
        for (int l = 0; l < BLOCK_SIZE; ++l) {
          val = min(val, shared_row[p][y][l] + shared_col[q][l][x]);
        }
        graph[i * n + j] = val;
      }
    }
  }
}

void apsp(int n, /* device */ int *graph) {
  int rounds = (n - 1) / BLOCK_SIZE + 1;
  int rounds_phase2 = (n - 1) / (BLOCK_SIZE * BATCH_SIZE_PHASE2) + 1;
  int rounds_phase3 = (n - 1) / (BLOCK_SIZE * BATCH_SIZE_PHASE3) + 1;
  dim3 thr(BLOCK_SIZE, BLOCK_SIZE);
  dim3 blk_phase2(rounds_phase2, 2);
  dim3 blk_phase3(rounds_phase3, rounds_phase3);

  for (int k = 0; k < rounds; ++k) {
    phase1<<<1, thr>>>(n, k, graph);
    phase2<<<blk_phase2, thr>>>(n, k, graph);
    phase3<<<blk_phase3, thr>>>(n, k, graph);
  }
}
