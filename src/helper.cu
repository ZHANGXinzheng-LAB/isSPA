#include <cstdio>
#include <fstream>

#include <cuda_runtime.h>

#include "helper.cuh"

int GetDeviceCount() 
{
    // 获取GPU数量
    int devcount{};
    cudaGetDeviceCount(&devcount);

    return devcount;
}

void DeviceMemoryUsage() 
{
    size_t free{};
    size_t total{};
    cudaMemGetInfo(&free, &total);
    std::printf("Device memory total: %zu MB, free: %zu MB, used: %zu MB\n", total >> 20, free >> 20, (total - free) >> 20); // 其中右移算符可以认为是除以1024*1024，因此得到MB
}

std::vector<std::pair<int, int>> work_intervals(int first, int last, int processor_count) 
{
    std::vector<std::pair<int, int>> ret;
    float total_works = last - first;
    int works_per_processor = std::max(static_cast<int>(std::round(total_works / processor_count)), 1);
    for (int i = 0; i < processor_count && first < last; ++i) 
    {
        ret.push_back({first, std::min(first + works_per_processor, last)});
        first += works_per_processor;
    }
    ret.back().second = last;
    return ret;
}

void dump(const std::string& filename, const void* ptr, size_t size) {
  std::fstream output(filename, std::ios::binary | std::ios::out | std::ios::trunc);
  auto pb = reinterpret_cast<const char*>(ptr);
  output.write(pb, size);
}