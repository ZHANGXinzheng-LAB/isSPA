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

    int padding_size, padded_template_size, image_size;
    size_t batch_size;
    int nimg;
    size_t N_pixel;
    int grid_size, N_pixel1, grid_size1; 
    int ft_padding, ft_padded_size;
    size_t N_pixel2;
    int grid_size2, N_pixel3, grid_size3; 
    int overlap;
    int nx, ny;
    int block_x, block_y;
    int total_line_count, bin;
    float phi_step;
    bool invert;
    bool image_dependent_allocated;
    int fsc_size, nk_size;

    SearchNorm(const Config & c, const EulerData & e, Size img, const TextFileData & text_data, int device = 0, int fourier_pad = 0);
    ~SearchNorm();

    void LoadTemplate(const Templates & temp);
    void LoadImage(const Image & img);
    void SetParams(const Image::Params & params);
    void PreprocessTemplate();
    void PreprocessImage(const Image & img);
    void PreprocessImage(const Image & img, const std::vector<float> & k, const std::vector<float> & fsc);
    void SplitImage();
    void RotateTemplate(float euler3);
    void ComputeCCGSum();
    void ComputeCCGMean();
    void PickParticles(std::vector<float> & scores);
    void OutputScore(std::string & output, std::vector<float> & scores, float euler3, const Image & input, int & line_count);
    void OutputScore(std::string & output, std::vector<float> & scores, const Image & input);

    void work_verbose(const Image & image, std::string & output);
    void saveComplexToBinary(const cufftComplex* data, size_t size, const std::string& filename);
    void WriteTemplates(const std::string & filename, int nx, int ny, int nz, float pixel_size);

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