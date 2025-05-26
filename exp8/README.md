# 小作业八：性能分析实践

这个小作业直接按照助教的指示来写就行了，下面的内容是写了玩的。

## [`main`](./main) 干了什么？

反编译之，发现其主要有三个函数：`main()`、`init()` 和 `lipsum()`。

借助相关大模型工具，可以还原出附录中的代码。

我们只关心这一段：

```cpp
unsigned int a = static_cast<unsigned int>(input_id);
unsigned int b = a;
for (int i = 0; i < 10; ++i) {
    a = a * 0x2717 + 7;
    b = b * 0x2717 + 9U ^ 1;
    a = a % 1000000007;
    b = b % 1000000007;
}
slowRank = a % 8;
workloadType = b % 2;
```

这段代码的作用是根据输入的学生 ID 计算出 `slowRank` 和 `workloadType`，从而决定了 MPI 程序的执行方式。

当我输入我的学号 `2022010650` 时，计算得到以下结果：

```plaintext
id = 2022010650
a = 684863513
b = 427617569
slowRank = 1
workloadType = 1
```

这意味着导致检测到的热点的进程的 MPI rank 是 `1`。

观察代码可以发现，`workloadType` 的值决定了 `lipsum()` 函数的执行方式：
- 当 `workloadType` 为 `0` 时，执行整数稀疏矩阵乘。
- 当 `workloadType` 为 `1` 时，执行浮点数稠密矩阵乘。

所以对我的学号而言，该程序中 lipsum 函数是浮点数稠密矩阵乘。

可以直接运行 [`task.py`](./task.py) 脚本来得到答案。

## 附录：反编译得到的完整代码

```cpp
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>

int mpiRank = 0;
int mpiSize = 0;
int id = -1;
int slowRank = 0;
int workloadType = 0;

const int CHUNK_SIZE = 5010;
const int NUM_CHUNKS = 4;
const int TOTAL_SIZE = CHUNK_SIZE * NUM_CHUNKS;

int ai[TOTAL_SIZE];
int bi[TOTAL_SIZE];
double ad[TOTAL_SIZE];
double bd[TOTAL_SIZE];
int ci[TOTAL_SIZE * CHUNK_SIZE];
double cd[TOTAL_SIZE * CHUNK_SIZE];

void init() {
    for (int chunk = 0; chunk < NUM_CHUNKS; ++chunk) {
        int offset = chunk * CHUNK_SIZE;
        for (int i = 0; i < CHUNK_SIZE; ++i) {
            ai[offset + i] = 12345;
            bi[offset + i] = 12345;
            ad[offset + i] = 1e6;
            bd[offset + i] = 1e6;
        }
    }
}

void lipsum() {
    if (workloadType == 0) {
        int baseIter = (mpiRank == slowRank) ? 6000 : 3000;
        int totalItems = (((id * 0x2717) % 100 + 0xaa) * baseIter) / 200;
        int itemCount = std::max(totalItems, 1);
        size_t blockSize = itemCount * sizeof(int);
        int* inputA = ai;
        int* output = ci;
        for (int outer = 0; outer < itemCount; ++outer) {
            std::memset(output, 0, blockSize);
            for (int row = 0; row < itemCount; ++row) {
                int acc = 0;
                int seed = 100;
                for (int col = 0; col < itemCount / 100; ++col) {
                    seed = (seed * 0x2717 + 0xd5) % itemCount;
                    int a = inputA[col];
                    int b = bi[seed * CHUNK_SIZE + col];
                    acc += a * b;
                }
                output[row] = acc;
            }
            inputA += CHUNK_SIZE;
            output = reinterpret_cast<int*>(
                reinterpret_cast<char*>(output) + CHUNK_SIZE * sizeof(int)
            );
        }
    } else {
        int baseIter = (mpiRank == slowRank) ? 2000 : 1000;
        int totalItems = (((id * 0x2717) % 100 + 0xaa) * baseIter) / 200;
        int itemCount = std::max(totalItems, 1);
        size_t blockSize = itemCount * sizeof(double);
        double* inputA = ad;
        double* output = cd;
        for (int outer = 0; outer < itemCount; ++outer) {
            std::memset(output, 0, blockSize);
            double* inputB = bd;
            for (int row = 0; row < itemCount; ++row) {
                double sum = 0.0;
                for (int col = 0; col < itemCount; ++col) {
                    double a = inputA[col];
                    double b = inputB[col];
                    sum += a * b;
                }
                output[row] = sum;
                inputB += CHUNK_SIZE;
            }
            inputA += CHUNK_SIZE;
            output = reinterpret_cast<double*>(
                reinterpret_cast<char*>(output) + CHUNK_SIZE * sizeof(double)
            );
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    if (mpiSize == 8) {
        if (argc == 2) {
            char* endptr = nullptr;
            long input_id = std::strtol(argv[1], &endptr, 10);
            id = static_cast<int>(input_id);
            if (id >= 0) {
                unsigned int a = static_cast<unsigned int>(input_id);
                unsigned int b = a;
                for (int i = 0; i < 10; ++i) {
                    a = a * 0x2717 + 7;
                    b = b * 0x2717 + 9U ^ 1;
                    a = a % 1000000007;
                    b = b % 1000000007;
                }
                slowRank = a % 8;
                workloadType = b % 2;
                init();
                MPI_Barrier(MPI_COMM_WORLD);
                lipsum();
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Finalize();
                if (mpiRank == 0) {
                    std::cout << "Success " << id << std::endl;
                }
                return 0;
            }
        }
        if (mpiRank == 0) {
            std::fprintf(stderr, "Usage: %s <Student ID>\n", argv[0]);
        }
    } else {
        if (mpiRank == 0) {
            std::fprintf(stderr, "Please run with %d MPI processes\n", 8);
        }
    }
    std::exit(1);
}
```
