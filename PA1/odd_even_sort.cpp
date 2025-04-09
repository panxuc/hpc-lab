#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#include "worker.h"

void radixSort(float *arr, int len) {
  const int RADIX = 256;
  uint32_t *src = new uint32_t[len];
  uint32_t *buf = new uint32_t[len];

  for (int i = 0; i < len; i++) {
    uint32_t x;
    std::memcpy(&x, &arr[i], sizeof(float));
    src[i] = (x & 0x80000000) ? ~x : (x ^ 0x80000000);
  }

  for (int pass = 0; pass < 4; pass++) {
    int count[RADIX] = {0};
    int shift = pass * 8;

    for (int i = 0; i < len; i++) {
      int bucket = (src[i] >> shift) & 0xFF;
      count[bucket]++;
    }
    int total = 0;
    for (int i = 0; i < RADIX; i++) {
      int oldCount = count[i];
      count[i] = total;
      total += oldCount;
    }
    for (int i = 0; i < len; i++) {
      int bucket = (src[i] >> shift) & 0xFF;
      buf[count[bucket]++] = src[i];
    }
    std::swap(src, buf);
  }

  for (int i = 0; i < len; i++) {
    uint32_t x = src[i];
    uint32_t y = (x & 0x80000000) ? (x ^ 0x80000000) : ~x;
    float f;
    std::memcpy(&f, &y, sizeof(float));
    arr[i] = f;
  }

  delete[] src;
  delete[] buf;
}

void mergeArrays(float *arr1, int len1, float *arr2, int len2, float *out) {
  float *end1 = arr1 + len1;
  float *end2 = arr2 + len2;
  while (true) {
    if (arr1 == end1) {
      memcpy(out, arr2, sizeof(float) * (end2 - arr2));
      return;
    }
    if (arr2 == end2) {
      memcpy(out, arr1, sizeof(float) * (end1 - arr1));
      return;
    }
    *out++ = (*arr2 < *arr1) ? *arr2++ : *arr1++;
  }
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  if (out_of_range)
    return;
  if (block_len < 100) {
    std::sort(data, data + block_len);
  } else {
    radixSort(data, block_len);
  }
  if (nprocs == 1)
    return;

  int rounds = nprocs;
  float *partnerBuffer = new float[block_len];
  float *mergedBuffer = new float[block_len * 2];

  MPI_Request request;
  MPI_Status status;

  for (int round = 0; round < rounds; round++) {
    int partner = -1;
    if (round % 2 == 0) {
      if (rank % 2 == 0)
        partner = rank + 1;
      else
        partner = rank - 1;
    } else {
      if (rank % 2 == 0)
        partner = rank - 1;
      else
        partner = rank + 1;
    }

    if (partner < 0 || partner >= nprocs) {
      continue;
    }

    MPI_Isend(data, block_len, MPI_FLOAT, partner, round, MPI_COMM_WORLD,
              &request);
    MPI_Recv(partnerBuffer, block_len, MPI_FLOAT, partner, round,
             MPI_COMM_WORLD, &status);
    MPI_Wait(&request, &status);

    mergeArrays(data, block_len, partnerBuffer, block_len, mergedBuffer);

    if (rank < partner) {
      std::memcpy(data, mergedBuffer, block_len * sizeof(float));
    } else {
      std::memcpy(data, mergedBuffer + block_len, block_len * sizeof(float));
    }
  }

  free(partnerBuffer);
  free(mergedBuffer);
}
