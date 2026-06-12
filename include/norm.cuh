#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <memory>
#include <ostream>
#include <string>

#include "DataReader2.h"
#include "emdata.h"
#include "image.hpp"
#include "templates.hpp"

struct SearchNorm 
{
    Parameters para;
    EulerData euler;

    struct Size 
    {
        size_t width;
        size_t height;
    };

    struct impl;
    std::unique_ptr<impl> pimpl;

    std::vector<int> block_offsets_x, block_offsets_y;

    int padded_template_size, image_size;
    int batch_size;
    int N_pixel, grid_size, N_pixel1, grid_size1, nimg;
    int padding_size;
    int overlap;
    int nx, ny;
    int block_x, block_y;
    int line_count, bin;
    float phi_step;
    bool invert, phase_flip;
    bool image_dependent_allocated;

    SearchNorm(const Config & c, const EulerData & e, Size img, int device = 0);
    ~SearchNorm();

    void LoadTemplate(const Templates & temp);
    void LoadImage(const Image & img);
    void SetParams(const Image::Params & params);
    void PreprocessTemplate(const std::vector<float> & k, const std::vector<float> & fsc);
    void PreprocessTemplate();
    void PreprocessImage(const Image & img);
    void SplitImage();
    void RotateTemplate(float euler3);
    void ComputeCCGSum();
    void ComputeCCGMean();
    void PickParticles(std::vector<float> & scores, float euler3);
    void OutputScore(std::string & output, std::vector<float> & scores, float euler3, const Image & input);

    void work_verbose(const Templates & temp, const Image & image, std::string & output, const std::vector<float> & k, const std::vector<float> & fsc);
    void work_verbose(const Templates & temp, const Image & image, std::string & output);
    void saveComplexToBinary(const cufftComplex* data, size_t size, const std::string& filename);

};

template <typename T>
void saveDataToFile(const T * data, size_t size, const std::string & filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        return;
    }
    for (size_t i = 0; i < size; i++)
    {
        file << data[i] << '\n';
    }

    file.close();
    std::cout << "Data has been saved to " << filename << std::endl;
}