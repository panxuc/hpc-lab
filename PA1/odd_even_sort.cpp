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

void mergeArrays(const float *arr1, int len1, const float *arr2, int len2, float *out) {
  int i = 0, j = 0, k = 0;
  while (i < len1 && j < len2) {
    if (arr1[i] <= arr2[j]) {
      out[k++] = arr1[i++];
    } else {
      out[k++] = arr2[j++];
    }
  }
  if (i < len1) {
    std::memcpy(out + k, arr1 + i, (len1 - i) * sizeof(float));
  }
  if (j < len2) {
    std::memcpy(out + k, arr2 + j, (len2 - j) * sizeof(float));
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
  float* partnerBuffer = new float[block_len];
  float* mergedBuffer  = new float[block_len * 2];

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

    MPI_Status status;
    MPI_Sendrecv(data, block_len, MPI_FLOAT, partner, round,
                 partnerBuffer, block_len, MPI_FLOAT, partner, round,
                 MPI_COMM_WORLD, &status);

    mergeArrays(data, block_len, partnerBuffer, block_len, mergedBuffer);

    if (rank < partner) {
      std::memcpy(data, mergedBuffer, block_len * sizeof(float));
    } else {
      std::memcpy(data, mergedBuffer + block_len, block_len * sizeof(float));
    }
  }

  delete[] partnerBuffer;
  delete[] mergedBuffer;
}
