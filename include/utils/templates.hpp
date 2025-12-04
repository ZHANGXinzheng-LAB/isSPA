#pragma once

#include <memory>

#include "DataReader2.h"
#include "emdata.h"

struct Templates 
{
    Templates() = default;
    Templates(const std::string & path, size_t cnt);

    size_t count; // 模板个数
    size_t width;
    size_t height;
    size_t bytes;
    std::unique_ptr<float[]> data;

    mrcH * mrch;
};