#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

float frand(float lo, float hi);
float grand(float mean, float sigma);
bool is_big_endian();
bool is_little_endian();

#define INIT_TIMEIT()                             \
  auto start = std::chrono::steady_clock().now(); \
  auto end = start;                               \
  std::chrono::milliseconds dt_ms {}
#define TIMEIT(F)                                                             \
  start = std::chrono::steady_clock().now();                                  \
  F;                                                                          \
  end = std::chrono::steady_clock().now();                                    \
  dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start); \
  std::cout << dt_ms.count() << " ms" << std::endl
#define CHECK()                                                                                   \
  if ((cudaPeekAtLastError()) != cudaSuccess) {                                                   \
    std::printf("\"%s\" at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
    std::exit(-1);                                                                                \
  }
#define CALL(F)                                                                                \
  if ((F) != cudaSuccess) {                                                                    \
    printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), __FILE__, __LINE__); \
    exit(-1);                                                                                  \
  }
#define FFTCALL(F)                                           \
  if ((F) != CUFFT_SUCCESS) {                                \
    printf("Error id:%d at %s:%d\n", F, __FILE__, __LINE__); \
    exit(-1);                                                \
  }

inline int portable_fseek(FILE *fp, off_t offset, int whence) {
#if defined(HAVE_FSEEKO)
  return fseeko(fp, offset, whence);
#elif defined(HAVE_FSEEK64)
  return fseek64(fp, offset, whence);
#elif defined(__BEOS__)
  return _fseek(fp, offset, whence);
#else
  return fseek(fp, offset, whence);
#endif
}
