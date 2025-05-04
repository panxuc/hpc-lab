// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"

#define BLOCK_SIZE 32
#define OFFSET (BLOCK_SIZE * BLOCK_SIZE)
#define BATCH_SIZE_PHASE2 6
#define BATCH_SIZE_PHASE3 6

namespace {

__global__ void phase1(int n, int k, int *graph) {
    __shared__ int shared[BLOCK_SIZE][BLOCK_SIZE];
    int x = threadIdx.x;
    int y = threadIdx.y;
    int i = k * BLOCK_SIZE + y;
    int j = k * BLOCK_SIZE + x;

    if (i < n && j < n)
        shared[y][x] = graph[i * n + j];
    else
        shared[y][x] = INT_MAX / 2;
    __syncthreads();

    int reg_shared_yx = shared[y][x];
    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int val = shared[y][k] + shared[k][x];
        if (val < reg_shared_yx)
            reg_shared_yx = val;
    }
    shared[y][x] = reg_shared_yx;

    if (i < n && j < n)
        graph[i * n + j] = shared[y][x];
}

__global__ void phase2(int n, int k, int *graph) {
    __shared__ int shared_pivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_block[BLOCK_SIZE][BLOCK_SIZE];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int bidx = blockIdx.x;
    if (bidx == k) return;

    // Process row blocks
    int i = k * BLOCK_SIZE + y;
    int j = bidx * BLOCK_SIZE + x;
    if (i < n && j < n)
        shared_block[y][x] = graph[i * n + j];
    else
        shared_block[y][x] = INT_MAX / 2;

    int pivot_i = k * BLOCK_SIZE + y;
    int pivot_j = k * BLOCK_SIZE + x;
    if (pivot_i < n && pivot_j < n)
        shared_pivot[y][x] = graph[pivot_i * n + pivot_j];
    else
        shared_pivot[y][x] = INT_MAX / 2;

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int val = shared_pivot[y][k] + shared_block[k][x];
        if (val < shared_block[y][x])
            shared_block[y][x] = val;
        __syncthreads();
    }

    if (i < n && j < n)
        graph[i * n + j] = shared_block[y][x];

    // Process column blocks
    i = bidx * BLOCK_SIZE + y;
    j = k * BLOCK_SIZE + x;
    if (i < n && j < n)
        shared_block[y][x] = graph[i * n + j];
    else
        shared_block[y][x] = INT_MAX / 2;

    pivot_i = k * BLOCK_SIZE + y;
    pivot_j = k * BLOCK_SIZE + x;
    if (pivot_i < n && pivot_j < n)
        shared_pivot[y][x] = graph[pivot_i * n + pivot_j];
    else
        shared_pivot[y][x] = INT_MAX / 2;

    __syncthreads();

    for (int k = 0; k < BLOCK_SIZE; ++k) {
        int val = shared_block[y][k] + shared_pivot[k][x];
        if (val < shared_block[y][x])
            shared_block[y][x] = val;
        __syncthreads();
    }

    if (i < n && j < n)
        graph[i * n + j] = shared_block[y][x];
}

__global__ void phase3(int n, int k, int *graph) {
    __shared__ int shared_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int shared_col[BLOCK_SIZE][BLOCK_SIZE];

    int x = threadIdx.x;
    int y = threadIdx.y;
    int block_i = blockIdx.y;
    int block_j = blockIdx.x;

    if (block_i == k || block_j == k) return;

    int i = block_i * BLOCK_SIZE + y;
    int j = block_j * BLOCK_SIZE + x;

    int row_i = block_i * BLOCK_SIZE + y;
    int row_j = k * BLOCK_SIZE + x;

    int col_i = k * BLOCK_SIZE + y;
    int col_j = block_j * BLOCK_SIZE + x;

    if (row_i < n && row_j < n)
        shared_row[y][x] = graph[row_i * n + row_j];
    else
        shared_row[y][x] = INT_MAX / 2;

    if (col_i < n && col_j < n)
        shared_col[y][x] = graph[col_i * n + col_j];
    else
        shared_col[y][x] = INT_MAX / 2;

    __syncthreads();

    int dist = (i < n && j < n) ? graph[i * n + j] : INT_MAX / 2;

    for (int k = 0; k < BLOCK_SIZE; ++k) {
        dist = min(dist, shared_row[y][k] + shared_col[k][x]);
    }

    if (i < n && j < n)
        graph[i * n + j] = dist;
}

}

void apsp(int n, /* device */ int *graph) {
    int rounds = (n - 1) / BLOCK_SIZE + 1;
    // int rounds_phase2 = (n - 1) / (BLOCK_SIZE * BATCH_SIZE_PHASE2) + 1;
    // int rounds_phase3 = (n - 1) / (BLOCK_SIZE * BATCH_SIZE_PHASE3) + 1;
    dim3 thr(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 blk_phase2(rounds_phase2, 2);
    // dim3 blk_phase3(rounds_phase3, rounds_phase3);
    dim3 blk_phase2(rounds);
    dim3 blk_phase3(rounds, rounds);

    for (int k = 0; k < rounds; ++k) {
        phase1<<<1, thr>>>(n, k, graph);
        phase2<<<blk_phase2, thr>>>(n, k, graph);
        phase3<<<blk_phase3, thr>>>(n, k, graph);
    }
}
