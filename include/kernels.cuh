#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include "constants.h"
#include "DataReader2.h"

__global__ void UpdateSigma(cufftComplex* d_templates, float* d_buf, const int N);
__global__ void generate_mask(int l, cufftComplex * mask, float r, float * d_buf, float up, float low);
__global__ void multiCount_dot(int l, cufftComplex* mask, cufftComplex* d_templates, float* constants, float* res);
__global__ void scale_each(int l, cufftComplex* d_templates, float * d_sigmas, const int N);
__global__ void SQRSum_by_circle(cufftComplex * data, float * ra, float * rb, int nx, int ny, const unsigned int N);
__global__ void whiten_Tmp(cufftComplex* data, float* ra, float* rb, int l, const unsigned int N);
__global__ void whiten_filter_weight_Img(cufftComplex * data, float * ra, float * rb, int nx, int ny, Parameters para, float * k, float * fsc, int size, float * k1, float * nk, int size1);
__global__ void whiten_filter_weight_Img(cufftComplex* data, float* ra, float* rb, int nx, int ny, Parameters para);
__global__ void whiten_filter_weight_Img(cufftComplex * data, float * ra, float * rb, int nx, int ny, Parameters para, float * k, float * fsc, int size, int mode);
__global__ void set_0Hz_to_0_at_RI(cufftComplex* data);
__global__ void apply_mask(cufftComplex* data, float d_m, float edge_half_width, int l, const unsigned int N);
__global__ void apply_weighting_function(cufftComplex * data, int l, Parameters para, const unsigned int N);
__global__ void compute_area_sum_ofSQR(cufftComplex * data, float * res, int l, const int N);
__global__ void compute_sum_sqr(cufftComplex* data, float* res, const unsigned int N);
__global__ void normalize(cufftComplex * data, int image_size, float * means, const int N);
__global__ void divided_by_var(cufftComplex * data, int image_size, float * var, const unsigned int N);
__global__ void substract_by_mean(cufftComplex * data, int image_size, float * means, const unsigned int N);
__global__ void rotate_IMG(float* d_image, float* d_rotated_image, float e, int nx, int ny);
__global__ void rotate_subIMG(cufftComplex* d_image, cufftComplex* d_rotated_image, float e, int l, const unsigned int N);
__global__ void split_IMG(float* Ori, cufftComplex* IMG, int nx, int ny, int l, int bx, int overlap);
__global__ void split_IMG(float* Ori, cufftComplex* IMG, int* block_off_x, int* block_off_y, int nx, int ny, int l, int bx, const int N);
__global__ void compute_corner_CCG(cufftComplex* CCG, cufftComplex* Tl, cufftComplex* IMG, int l, int block_id, const unsigned int N);
__global__ void add_CCG_to_sum(cufftComplex* CCG_sum, cufftComplex* CCG, int l, int N_tmp, int block_id);
__global__ void set_CCG_mean(cufftComplex * CCG_sum, int N, int total_n);
__global__ void update_CCG(cufftComplex* CCG_sum, cufftComplex* CCG, int image_size, int block_id, unsigned int N);
__global__ void get_peak_and_SUM(cufftComplex* odata, float* res, int l, float d_m);
__global__ void get_peak_pos(cufftComplex* odata, float* res, int image_size, const unsigned int N);
__global__ void fft_shift_pad(cufftComplex * output, cufftComplex * input, int l, int l0, const int N, int width=8);
__global__ void scale(cufftComplex* data, int size, int l2);
__global__ void ri2ap(cufftComplex* data, size_t size);
__global__ void ap2ri(cufftComplex* data, unsigned int N);
__global__ void Complex2float(float* f, cufftComplex* c, int N);
__global__ void float2Complex(cufftComplex* c, float* f, int N);
__global__ void do_phase_flip(cufftComplex* filter, Parameters para, int nx, int ny);
__device__ float CTF_AST(int x1, int y1, int nx, int ny, float apix, float dfu, float dfv, float dfdiff, float dfang, float lambda, float cs, float ampconst, int mode);
__device__ float CTF(int x1, int y1, int nx, int ny, float apix, float dfu, float dfv, float dfdiff, float dfang, float lambda, float cs, float ampconst, int mode);
__device__ float interp_fsc(const float* keys, const float* values,
    int size, const float query);
__device__ float interp_nk(const float* keys, const float* values,
    int size, const float query);
