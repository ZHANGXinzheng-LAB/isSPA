#pragma once

#include <memory>
#include <string>

#include "DataReader2.h"

struct Image 
{
    struct Params 
    {
        double defocus;
        double dfdiff;
        double dfang;
        size_t width;
        size_t height;
    };

    // constructor
    Image(const LST::Entry & input);

    std::unique_ptr<float[]> data;
    std::string rpath;
    Params p;
    int unused;
};