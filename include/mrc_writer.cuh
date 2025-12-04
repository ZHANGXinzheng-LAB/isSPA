// mrc_writer.h
#ifndef MRC_WRITER_H
#define MRC_WRITER_H

#include <cufft.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

#pragma pack(push, 1) // 禁用内存对齐，确保头结构体大小为1024字节
struct MRCHeader {
    // 基础字段 ---------------------------------------------------------------
    int32_t nx, ny, nz;          // 0-11   三维尺寸 (X,Y,Z)
    int32_t mode;                // 12-15  数据类型: 0=int8,1=int16,2=float32,6=uint16
    int32_t nxstart, nystart, nzstart; // 16-27 起始索引 (通常为0)
    int32_t mx, my, mz;          // 28-39   网格数 (通常等于nx,ny,nz)
    float cella[3];              // 40-51   晶胞尺寸 (Å)
    float cellb[3];              // 52-63   晶胞角度 (°)
    int32_t mapc, mapr, maps;    // 64-75   坐标轴映射 (1=X,2=Y,3=Z)
    float dmin, dmax, dmean;     // 76-95   最小、最大、平均密度值
    int32_t ispg;                // 96-99   空间群编号 (电镜数据通常为0或1)
    int32_t nsymbt;              // 100-103 扩展头字节数 (通常为0)
    int32_t extra[25];             // 104-203 保留字段
    float origin[3];             // 204-215 原点坐标 (Å)
    char map[4];                 // 216-219 标识符 "MAP "
    int32_t machst;              // 220-223 机器标识 (0x44 0x44 0x00 0x00 表示小端序)
    float rms;                  // 224-227 RMS偏差
    int32_t nlabl;               // 228-231 标签数 (最多10)
    char labels[800];            // 232-1031 标签文本 (每标签80字符)
};
#pragma pack(pop)

// 三维体积写入函数
void write_mrc(
    const std::string& filename,
    const float* volume_data,    // 输入数据指针 [z][y][x]
    int nx, int ny, int nz,      // 三维尺寸
    float pixel_size)    // 像素尺寸 (Å)
{
    // 1. 准备头文件 ----------------------------------------------------------
    MRCHeader header = {};
    
    // 基础尺寸
    header.nx = nx;
    header.ny = ny;
    header.nz = nz;
    header.mx = nx;
    header.my = ny;
    header.mz = nz;

    header.nxstart = 0;
    header.nystart = 0;
    header.nzstart = 0;

    // 数据类型 (假设输入为float32)
    header.mode = 2;

    // 晶胞参数 (假设各向同性)
    header.cella[0] = nx * pixel_size;
    header.cella[1] = ny * pixel_size;
    header.cella[2] = nz * pixel_size;
    header.cellb[0] = 90.0f; // alpha
    header.cellb[1] = 90.0f; // beta
    header.cellb[2] = 90.0f; // gamma

    // 坐标映射 (标准: X=1,Y=2,Z=3)
    header.mapc = 1;
    header.mapr = 2;
    header.maps = 3;

    // 机器标识 (小端序)
    header.machst = 0x00004444;

    // 标识符
    memcpy(header.map, "MAP ", 4);

    // 自动计算统计值
    double sum = 0.0f;
    float dmin = INFINITY;
    float dmax = -INFINITY;
    const size_t total = nx * ny * nz;
    for (size_t i = 0; i < total; i++) 
    {
        float val = volume_data[i];
        sum += (double)val;
        if (val < dmin) dmin = val;
        if (val > dmax) dmax = val;
    }
    header.dmin = dmin;
    header.dmax = dmax;
    header.dmean = sum / total;

    // 其他字段
    header.ispg = 0;      // 非晶体数据
    header.nsymbt = 0;    // 无扩展头
    header.nlabl = 0;     // 无标签

    // 2. 写入文件 ------------------------------------------------------------
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("无法创建文件: " + filename);

    // 写入头
    file.write(reinterpret_cast<char*>(&header), sizeof(MRCHeader));

    // 写入数据 (按ZYX顺序)
    const size_t data_size = nx * ny * nz * sizeof(float);
    file.write(reinterpret_cast<const char*>(volume_data), data_size);

    if (!file.good()) throw std::runtime_error("写入文件失败: " + filename);
}

void write_mrc(
    const std::string& filename,
    const cufftComplex* volume_data,    // 输入数据指针 [z][y][x]
    int nx, int ny, int nz,      // 三维尺寸
    float pixel_size)    // 像素尺寸 (Å)
{
    // 1. 准备头文件 ----------------------------------------------------------
    MRCHeader header = {};
    
    // 基础尺寸
    header.nx = nx;
    header.ny = ny;
    header.nz = nz;
    header.mx = nx;
    header.my = ny;
    header.mz = nz;

    header.nxstart = 0;
    header.nystart = 0;
    header.nzstart = 0;

    // 数据类型 (假设输入为float32)
    header.mode = 2;

    // 晶胞参数 (假设各向同性)
    header.cella[0] = nx * pixel_size;
    header.cella[1] = ny * pixel_size;
    header.cella[2] = nz * pixel_size;
    header.cellb[0] = 90.0f; // alpha
    header.cellb[1] = 90.0f; // beta
    header.cellb[2] = 90.0f; // gamma

    // 坐标映射 (标准: X=1,Y=2,Z=3)
    header.mapc = 1;
    header.mapr = 2;
    header.maps = 3;

    // 机器标识 (小端序)
    header.machst = 0x00004444;

    // 标识符
    memcpy(header.map, "MAP ", 4);

    // 自动计算统计值
    double sum = 0.0f;
    float dmin = INFINITY;
    float dmax = -INFINITY;
    const size_t total = nx * ny * nz;
    for (size_t i = 0; i < total; i++) 
    {
        float val = volume_data[i].x;
        sum += val;
        if (val < dmin) dmin = val;
        if (val > dmax) dmax = val;
    }
    header.dmin = dmin;
    header.dmax = dmax;
    header.dmean = sum / total;

    // 其他字段
    header.ispg = 0;      // 非晶体数据
    header.nsymbt = 0;    // 无扩展头
    header.nlabl = 0;     // 无标签

    // 2. 写入文件 ------------------------------------------------------------
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("无法创建文件: " + filename);

    // 写入头
    file.write(reinterpret_cast<char*>(&header), sizeof(MRCHeader));

    // 写入数据 (按ZYX顺序)
    const size_t data_size = nx * ny * nz * sizeof(float);
    file.write(reinterpret_cast<const char*>(volume_data), data_size);

    if (!file.good()) throw std::runtime_error("写入文件失败: " + filename);
}

#endif // MRC_WRITER_H