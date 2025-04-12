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
      std::memcpy(out, arr2, sizeof(float) * (end2 - arr2));
      return;
    }
    if (arr2 == end2) {
      std::memcpy(out, arr1, sizeof(float) * (end1 - arr1));
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

  size_t block_size = ceiling(n, nprocs);
  const int len1 = (block_len + 1) / 2;
  const int len2 = block_size / 2;
  const int sz = block_size % 2 ? nprocs + nprocs / 2 : nprocs;
  int left = rank - 1;
  int right = rank + 1;
  float *recvbuf = new float[len2];
  float *sendbuf = new float[len1 + len2];

  for (int i = 0; i < sz; i++) {
    if (!rank) {
      // first rank
      std::memcpy(sendbuf + len2, data, sizeof(float) * len1);
      MPI_Sendrecv(data + len1, len2, MPI_FLOAT, right, rank, recvbuf, len2,
                   MPI_FLOAT, right, right, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (last_rank) {
      // last rank
      MPI_Sendrecv(data, len1, MPI_FLOAT, left, rank, recvbuf, len2, MPI_FLOAT,
                   left, left, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      mergeArrays(recvbuf, len2, data, len1, sendbuf);
    } else {
      // middle ranks
      MPI_Sendrecv(data + len1, len2, MPI_FLOAT, right, rank, recvbuf, len2,
                   MPI_FLOAT, left, left, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      mergeArrays(recvbuf, len2, data, len1, sendbuf);
      MPI_Sendrecv(sendbuf, len2, MPI_FLOAT, left, rank, recvbuf, len2,
                   MPI_FLOAT, right, right, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    mergeArrays(sendbuf + len2, len1, recvbuf, block_len - len1, data);
  }

  delete[] recvbuf;
  delete[] sendbuf;
}
