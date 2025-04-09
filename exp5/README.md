# 小作业五：自动向量化与基于 intrinsic 的手动向量化

## baseline, auto simd 和 intrinsic 版本的 a+b 的运行时间

| 版本      | 运行时间 |
| --------- | -------- |
| baseline  | 4437 us  |
| auto simd | 526 us   |
| intrinsic | 526 us   |

使用 intel intrinsics 填写 `aplusb-intrinsic.cpp` 中的函数后，通过结果检查，intrisic 版本的运行时间与 auto simd 版本相近。

## `a_plus_b_intrinsic` 函数的实现代码

```cpp
void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    // Your code here
    for (int i = 0; i < n; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_store_ps(c + i, vc);
    }
}
```
