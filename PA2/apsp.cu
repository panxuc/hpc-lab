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
  int val = MAX_VAL;
  for (int k = 0; k < BLOCK_SIZE; k++) {
    val = min(val, shared[y][k] + shared[k][x]);
  }
  if (i < n && j < n) {
    graph[i * n + j] = val;
  }
}

__global__ void phase2(int n, int k, int *graph) {
  __shared__ int shared_pivot[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int shared_block[BATCH_SIZE_PHASE2][BLOCK_SIZE][BLOCK_SIZE];
  int x = threadIdx.x;
  int y = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int pi = k * BLOCK_SIZE + y;
  int pj = k * BLOCK_SIZE + x;
  // int bi = by * BATCH_SIZE_PHASE2 * BLOCK_SIZE;
  int bj = bx * BATCH_SIZE_PHASE2 * BLOCK_SIZE;
  bool flag = (by == 0);
  int val = MAX_VAL;
  shared_pivot[y][x] = (pi < n && pj < n) ? graph[pi * n + pj] : MAX_VAL;
  if (flag) {
    for (int p = 0, j = bj; p < BATCH_SIZE_PHASE2; p++, j += BLOCK_SIZE) {
      shared_block[p][y][x] =
          (pi < n && j + x < n) ? graph[pi * n + j + x] : MAX_VAL;
    }
  } else {
    for (int p = 0, i = bj; p < BATCH_SIZE_PHASE2; p++, i += BLOCK_SIZE) {
      shared_block[p][y][x] =
          (i + y < n && pj < n) ? graph[(i + y) * n + pj] : MAX_VAL;
    }
  }
  __syncthreads();
  if (flag) {
    for (int p = 0, j = bj; p < BATCH_SIZE_PHASE2; p++, j += BLOCK_SIZE) {
      val = MAX_VAL;
      for (int k = 0; k < BLOCK_SIZE; k++) {
        val = min(val, shared_pivot[y][k] + shared_block[p][k][x]);
      }
      if (pi < n && j + x < n) {
        graph[pi * n + j + x] = val;
      }
    }
  } else {
    for (int p = 0, i = bj; p < BATCH_SIZE_PHASE2; p++, i += BLOCK_SIZE) {
      val = MAX_VAL;
      for (int k = 0; k < BLOCK_SIZE; k++) {
        val = min(val, shared_block[p][y][k] + shared_pivot[k][x]);
      }
      if (i + y < n && pj < n) {
        graph[(i + y) * n + pj] = val;
      }
    }
  }
}

__global__ void phase3(int n, int k, int *graph) {
  __shared__ int shared_row[BATCH_SIZE_PHASE3][BLOCK_SIZE][BLOCK_SIZE];
  __shared__ int shared_col[BATCH_SIZE_PHASE3][BLOCK_SIZE][BLOCK_SIZE];
  int x = threadIdx.x;
  int y = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int pi = k * BLOCK_SIZE + y;
  int pj = k * BLOCK_SIZE + x;
  int bi = by * BATCH_SIZE_PHASE3 * BLOCK_SIZE;
  int bj = bx * BATCH_SIZE_PHASE3 * BLOCK_SIZE;
  int val = MAX_VAL;
  for (int k = 0, j = bj; k < BATCH_SIZE_PHASE3; k++, j += BLOCK_SIZE) {
    shared_col[k][y][x] =
        (pi < n && j + x < n) ? graph[pi * n + j + x] : MAX_VAL;
  }
  for (int k = 0, i = bi; k < BATCH_SIZE_PHASE3; k++, i += BLOCK_SIZE) {
    shared_row[k][y][x] =
        (i + y < n && pj < n) ? graph[(i + y) * n + pj] : MAX_VAL;
  }
  __syncthreads();
  if (bi + BATCH_SIZE_PHASE3 * BLOCK_SIZE <= n &&
      bj + BATCH_SIZE_PHASE3 * BLOCK_SIZE <= n) {
    for (int p = 0, i = bi; p < BATCH_SIZE_PHASE3; p++, i += BLOCK_SIZE) {
      for (int q = 0, j = bj; q < BATCH_SIZE_PHASE3; q++, j += BLOCK_SIZE) {
        val = graph[(i + y) * n + j + x];
        for (int k = 0; k < BLOCK_SIZE; k++)
          val = min(val, shared_row[p][y][k] + shared_col[q][k][x]);
        graph[(i + y) * n + j + x] = val;
      }
    }
  } else {
    for (int p = 0, i = bi; p < BATCH_SIZE_PHASE3; p++, i += BLOCK_SIZE) {
      for (int q = 0, j = bj; q < BATCH_SIZE_PHASE3; q++, j += BLOCK_SIZE) {
        val = (i + y < n && j + x < n) ? graph[(i + y) * n + j + x] : MAX_VAL;
        for (int k = 0; k < BLOCK_SIZE; k++) {
          val = min(val, shared_row[p][y][k] + shared_col[q][k][x]);
        }
        if (i + y < n && j + x < n) {
          graph[(i + y) * n + j + x] = val;
        }
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
  for (int k = 0; k < rounds; k++) {
    phase1<<<1, thr>>>(n, k, graph);
    phase2<<<blk_phase2, thr>>>(n, k, graph);
    phase3<<<blk_phase3, thr>>>(n, k, graph);
  }
}
