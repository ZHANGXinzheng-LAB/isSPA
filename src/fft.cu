#include "fft.cuh"

cufftHandle MakeFFTPlan(int dim0, int dim1, unsigned int batch_size) 
{
  cufftHandle plan{};

  constexpr int rank = 2;      // 维数
  int n[rank] = {dim0, dim1};  // n*m
  int* inembed = n;            // 输入的数组size
  int istride = 1;             // 数组内数据连续，为1
  int idist = n[0] * n[1];     // 1个数组的内存大小
  int* onembed = n;            // 输出是一个数组的size
  int ostride = 1;             // 每点DFT后数据连续则为1
  int odist = n[0] * n[1];  // 输出第一个数组与第二个数组的距离，即两个数组的首元素的距离
  int batch = batch_size;  // 批量处理的批数

  // FFT handler for all templates
  cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);  // 针对多信号同时进行FFT

  return plan;
};