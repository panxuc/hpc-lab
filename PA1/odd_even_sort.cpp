#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#include "worker.h"

void mergeArrays(const float* arr1, int len1, const float* arr2, int len2, float* out) {
  int i = 0, j = 0, k = 0;
  while (i < len1 && j < len2) {
    if (arr1[i] <= arr2[j]) {
      out[k++] = arr1[i++];
    } else {
      out[k++] = arr2[j++];
    }
  }
  while (i < len1) {
    out[k++] = arr1[i++];
  }
  while (j < len2) {
    out[k++] = arr2[j++];
  }
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data
  if (out_of_range)
    return;
  std::sort(data, data + block_len);
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
