#include <cuda_runtime.h>
#include <cufft.h>
#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <thread>
#include <cstdio>

#include "fft.cuh"
#include "helper.cuh"
#include "kernels.cuh"
#include "nonorm.cuh"
#include "smartptr.cuh"

struct SearchNoNorm::impl {
  struct Device {
    struct FFT {
      cufftHandle templates;
      cufftHandle image;
      cufftHandle raw_image;
    } fft;

    cudaStream_t stream;

    device_unique_ptr<cufftComplex[]> padded_templates;
    device_unique_ptr<cufftComplex[]> CCG;

    device_unique_ptr<cufftComplex[]> padded_image;
    device_unique_ptr<cufftComplex[]> padded_rotated_image;

    device_unique_ptr<double[]> sigmas;
    device_unique_ptr<float[]> image;
    device_unique_ptr<float[]> ra;
    device_unique_ptr<float[]> rb;
    device_unique_ptr<float[]> reduction_buf;
    device_unique_ptr<float[]> means;

    int id;
  } dev;

  struct Host {
    pinned_unique_ptr<float[]> reduction_buf;
    pinned_unique_ptr<double[]> sigmas;

    std::unique_ptr<float[]> ubuf;
  } host;
};

SearchNoNorm::SearchNoNorm(const Config & c, const EulerData & e, Size img, int device)
    : para(c),
      euler(e),
      padding_size(c.geti("Window_size")),
      overlap(c.geti("Overlap")),
      invert(c.geti("Invert")),
      phase_flip(c.geti("Phase_flip")),
      phi_step(c.getf("Phi_step")),
      batch_size(e.size()),
      nx(img.width),
      ny(img.height),
      line_count(0),
      pimpl(std::make_unique<impl>()) 
{
  std::cout << "Thread " << std::this_thread::get_id() << " sets device " << device << "\n";
  pimpl->dev.id = device;
  cudaSetDevice(device);

  padded_template_size = padding_size * padding_size;
  // set nx = ny = tile_size, not using original image size
  nx = ny = tile_size;

  auto blocks_one_axis = [](int length, int padding, int overlap) -> int 
  {
    int offset = 0, blocks = 1;
    while (offset + padding < length) 
    {
      offset += padding - overlap;
      if (offset + padding >= length) 
      {
        offset = length - padding;
      }
      ++blocks;
    }
    return blocks;
  };

  // num of blocks in x, y axis
  block_x = blocks_one_axis(nx, padding_size, overlap);
  block_y = blocks_one_axis(ny, padding_size, overlap);
  std::printf("Split block_x: %d, block_y: %d\n", block_x, block_y);

  // N = max{num of tmp, num of subimgs}
  batch_size = std::max(batch_size, static_cast<size_t>(block_x * block_y));

  cudaStreamCreate(&pimpl->dev.stream);
  pimpl->dev.padded_templates =
      make_device_unique<cufftComplex[]>(padded_template_size * batch_size);
  pimpl->dev.CCG = make_device_unique<cufftComplex[]>(padded_template_size * batch_size);
  pimpl->host.sigmas = make_host_unique_pinned<double[]>(batch_size);
  pimpl->dev.sigmas = make_device_unique<double[]>(batch_size);
  pimpl->dev.ra = make_device_unique<float[]>(batch_size * RA_SIZE);
  pimpl->dev.rb = make_device_unique<float[]>(batch_size * RA_SIZE);
  auto buf_size = 4 * padded_template_size * batch_size / BLOCK_SIZE;
  pimpl->host.reduction_buf = make_host_unique_pinned<float[]>(buf_size);
  pimpl->dev.reduction_buf = make_device_unique<float[]>(buf_size);
  pimpl->dev.means = make_device_unique<float[]>(batch_size);
  pimpl->host.ubuf = std::make_unique<float[]>(4 * batch_size);

  pimpl->dev.fft.templates = MakeFFTPlan(padding_size, padding_size, batch_size);
  cufftSetStream(pimpl->dev.fft.templates, pimpl->dev.stream);

  pimpl->dev.image = make_device_unique<float[]>(nx * ny);
  pimpl->dev.padded_image =
      make_device_unique<cufftComplex[]>(block_x * block_y * padded_template_size);
  pimpl->dev.padded_rotated_image =
      make_device_unique<cufftComplex[]>(block_x * block_y * padded_template_size);

  pimpl->dev.fft.image = MakeFFTPlan(padding_size, padding_size, block_x * block_y);
  cufftSetStream(pimpl->dev.fft.image, pimpl->dev.stream);
  pimpl->dev.fft.raw_image = MakeFFTPlan(ny, nx, 1);
  cufftSetStream(pimpl->dev.fft.raw_image, pimpl->dev.stream);

  cudaStreamSynchronize(pimpl->dev.stream);
  CHECK();
  DeviceMemoryUsage();
}

SearchNoNorm::~SearchNoNorm() {
  cufftDestroy(pimpl->dev.fft.raw_image);
  cufftDestroy(pimpl->dev.fft.image);
  cufftDestroy(pimpl->dev.fft.templates);
  cudaStreamDestroy(pimpl->dev.stream);
}

void SearchNoNorm::LoadTemplate(const Templates& temp) {
  auto padded_templates = std::make_unique<cufftComplex[]>(padded_template_size * batch_size);
  std::memset(padded_templates.get(), 0, sizeof(cufftComplex) * padded_template_size * batch_size);

  // padding
  int sx = (padding_size - temp.width) / 2;
  int sy = (padding_size - temp.height) / 2;
  const size_t size = temp.width * temp.height;
#pragma omp parallel for
  for (int n = 0; n < batch_size; ++n) {
    for (int j = 0; j < temp.height; j++) {
      for (int i = 0; i < temp.width; i++) {
        size_t index = padded_template_size * n + (sy + j) * padding_size + (sx + i);
        float cur = temp.data[n * size + i + j * temp.width];
        padded_templates[index].x = cur;
      }
    }
  }

  cudaMemcpyAsync(pimpl->dev.padded_templates.get(), padded_templates.get(),
                  sizeof(cufftComplex) * padded_template_size * batch_size, cudaMemcpyHostToDevice,
                  pimpl->dev.stream);

  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNoNorm::LoadImage(const TileImages::Tile& tile) {
  if (invert) {
#pragma omp parallel for
    for (int i = 0; i < nx * ny; i++) tile.data[i] = -tile.data[i];
  }
  cudaMemcpyAsync(pimpl->dev.image.get(), tile.data.get(), sizeof(float) * nx * ny,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNoNorm::SetParams(const TileImages::Params& params) 
{
  // set params
  para.defocus = -params.defocus;
  para.dfang = params.dfang;
  para.dfdiff = params.dfdiff;
  para.dfu = -params.defocus + params.dfdiff;  // -defocus is minus, so abs(dfu) < abs(dfv)
  para.dfv = -params.defocus - params.dfdiff;
  para.lambda = 12.2643 / sqrt(para.energy * 1000.0 + 0.97848 * para.energy * para.energy);
  para.ds = 1 / (para.apix * padding_size);
}

void SearchNoNorm::PreprocessTemplate() {
  const int l = padding_size;
  const int nblocks = padded_template_size * batch_size / BLOCK_SIZE;

  float r = l / 2.f - 2.f;
  float up_bound{}, low_bound{};
  if (r > 1) {
    up_bound = (r + 1) * (r + 1);
    low_bound = (r - 1) * (r - 1);
  }

  cudaMemsetAsync(pimpl->dev.CCG.get(), 0, sizeof(cufftComplex) * padded_template_size * batch_size,
                  pimpl->dev.stream);
  cudaMemsetAsync(pimpl->dev.reduction_buf.get(), 0,
                  4 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  pimpl->dev.stream);

  // generate Mask, count number of non-zero digits
  generate_mask<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(
      l, pimpl->dev.CCG.get(), r, pimpl->dev.reduction_buf.get(), up_bound, low_bound);

  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  std::memset(pimpl->host.ubuf.get(), 0, batch_size * sizeof(float));

  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    pimpl->host.ubuf[id] += pimpl->host.reduction_buf[k];
  }

  // Calculate dot of mask and all templates
  cudaMemcpyAsync(pimpl->dev.means.get(), pimpl->host.ubuf.get(), sizeof(float) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);

  multiCount_dot<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(
      l, pimpl->dev.CCG.get(), pimpl->dev.padded_templates.get(), pimpl->dev.means.get(),
      pimpl->dev.reduction_buf.get());

  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  std::memset(pimpl->host.ubuf.get(), 0, batch_size * sizeof(float));

  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    pimpl->host.ubuf[id] += pimpl->host.reduction_buf[k];
  }

  UpdateSigma<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float) * 2, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), pimpl->dev.reduction_buf.get());
  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  2 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  // put em on GPU
  cudaMemcpyAsync(pimpl->dev.means.get(), pimpl->host.ubuf.get(), sizeof(float) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);

  std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));

  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    pimpl->host.ubuf[2 * id] += pimpl->host.reduction_buf[2 * k];
    // sum of value
    pimpl->host.ubuf[2 * id + 1] += pimpl->host.reduction_buf[2 * k + 1];
    // sum of value^2
  }

  for (int i = 0; i < batch_size; i++) {
    double mean = pimpl->host.ubuf[2 * i] / (double)(l * l);
    pimpl->host.sigmas[i] = std::sqrt(pimpl->host.ubuf[2 * i + 1] / (double)(l * l) - mean * mean);
    if (pimpl->host.sigmas[i] < 0 || !std::isfinite(pimpl->host.sigmas[i]))
      pimpl->host.sigmas[i] = 0;
  }

  // data[i]=(data[i]-em)/s;
  cudaMemcpyAsync(pimpl->dev.sigmas.get(), pimpl->host.sigmas.get(), sizeof(double) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  scale_each<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      l, pimpl->dev.padded_templates.get(), pimpl->dev.means.get(), pimpl->dev.sigmas.get());

  // **************************************************************
  // apply whitening filter and do ift
  // input: Padded IMAGE (Real SPACE)
  // output: IMAGE_whiten (Fourier SPACE in RI)
  // **************************************************************
  cudaMemsetAsync(pimpl->dev.ra.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);
  cudaMemsetAsync(pimpl->dev.rb.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);

  // Inplace FFT
  cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(),
               pimpl->dev.padded_templates.get(), CUFFT_FORWARD);

  // CUFFT will enlarge VALUE to N times. Restore it
  scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), padded_template_size * batch_size, padded_template_size);

  // Whiten at fourier space
  // contain ri2ap
  SQRSum_by_circle<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size,
      padding_size);

  // contain ap2ri
  whiten_Tmp<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size);

  cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(),
               pimpl->dev.padded_templates.get(), CUFFT_INVERSE);
  apply_mask<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), para.d_m, para.edge_half_width, padding_size);
  cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(),
               pimpl->dev.padded_templates.get(), CUFFT_FORWARD);
  // CUFFT will enlarge VALUE to N times. Restore it
  scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), padded_template_size * batch_size, padded_template_size);

  // **************************************************************
  // 1. lowpass
  // 2. apply weighting function
  // 3. normlize
  // input: masked_whiten_IMAGE (Fourier SPACE in RI)
  // output: PROCESSED_IMAGE (Fourier SPACE in AP)
  // **************************************************************
  // contain ri2ap
  apply_weighting_function<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), padding_size, para);

  compute_area_sum_ofSQR<<<nblocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float),
                           pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(),
                                                pimpl->dev.reduction_buf.get(), padding_size,
                                                padding_size);

  cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                  2 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                  cudaMemcpyDeviceToHost, pimpl->dev.stream);

  // After Reduction -> compute mean for each image
  std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));
  float* infile_mean = pimpl->host.ubuf.get();
  float* counts = pimpl->host.ubuf.get() + batch_size;
  cudaStreamSynchronize(pimpl->dev.stream);
  for (int k = 0; k < padded_template_size * batch_size / BLOCK_SIZE; k++) {
    int id = k / (padded_template_size / BLOCK_SIZE);
    infile_mean[id] += pimpl->host.reduction_buf[2 * k];
    counts[id] += pimpl->host.reduction_buf[2 * k + 1];
  }

  for (int k = 0; k < batch_size; k++) {
    infile_mean[k] = std::sqrt(infile_mean[k] / (counts[k] * counts[k]));
  }
  // Do Normalization with computed infile_mean[]
  cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * batch_size,
                  cudaMemcpyHostToDevice, pimpl->dev.stream);
  // Contain ap2ri
  normalize<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_templates.get(), padding_size, padding_size, pimpl->dev.means.get());
  cudaMemsetAsync(pimpl->dev.means.get(), 0, batch_size * sizeof(float), pimpl->dev.stream);

  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNoNorm::PreprocessImage() {
  if (phase_flip == 1) {
    int nblocks = std::ceil(nx * ny / static_cast<double>(BLOCK_SIZE));
    cudaMemsetAsync(pimpl->dev.padded_rotated_image.get(), 0,
                    block_x * block_y * padded_template_size * sizeof(cufftComplex),
                    pimpl->dev.stream);
    float2Complex<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
        pimpl->dev.padded_rotated_image.get(), pimpl->dev.image.get(), nx, ny);
    // fft inplace
    cufftExecC2C(pimpl->dev.fft.raw_image, pimpl->dev.padded_rotated_image.get(),
                 pimpl->dev.padded_rotated_image.get(), CUFFT_FORWARD);
    scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_rotated_image.get(),
                                                         nx * ny, nx * ny);

    // phase flip
    do_phase_flip<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
        pimpl->dev.padded_rotated_image.get(), para, nx, ny);

    // Whiten at fourier space
    cudaMemsetAsync(pimpl->dev.ra.get(), 0, batch_size * RA_SIZE * sizeof(float),
                    pimpl->dev.stream);
    cudaMemsetAsync(pimpl->dev.rb.get(), 0, batch_size * RA_SIZE * sizeof(float),
                    pimpl->dev.stream);

    // contain ri2ap
    SQRSum_by_circle<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
        pimpl->dev.padded_rotated_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, 1);

    // 1. whiten
    // 2. low pass
    // 3. weight
    // 4. ap2ri
    whiten_filter_weight_Img<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
        pimpl->dev.padded_rotated_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny,
        para);

    // ifft inplace
    cufftExecC2C(pimpl->dev.fft.raw_image, pimpl->dev.padded_rotated_image.get(),
                 pimpl->dev.padded_rotated_image.get(), CUFFT_INVERSE);
    Complex2float<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
        pimpl->dev.image.get(), pimpl->dev.padded_rotated_image.get(), nx, ny);
  }
  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNoNorm::SplitImage() {
  int l = padding_size;
  int nblocks = block_x * block_y * padded_template_size / BLOCK_SIZE;

  cudaMemsetAsync(pimpl->dev.padded_image.get(), 0, block_x * block_y * padded_template_size * sizeof(cufftComplex), pimpl->dev.stream);

  // split Image into blocks with overlap
  split_IMG<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.image.get(), pimpl->dev.padded_image.get(), nx, ny, padding_size, block_x, overlap);

  // do normalize to all subIMGs
  if (phase_flip == 1) 
  {
    // Inplace FFT
    cufftExecC2C(pimpl->dev.fft.image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(), CUFFT_FORWARD);
    // Scale IMG to normal size
    scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), block_x * block_y * padded_template_size, l * l);
    ri2ap<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(),
                                                         block_x * block_y * padded_template_size);
    compute_area_sum_ofSQR<<<nblocks, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float),
                             pimpl->dev.stream>>>(pimpl->dev.padded_image.get(),
                                                  pimpl->dev.reduction_buf.get(), l, l);

    cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                    4 * padded_template_size * batch_size / BLOCK_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, pimpl->dev.stream);

    int nimg = block_x * block_y;
    // After Reduction -> compute mean for each image
    float infile_mean[nimg], counts[nimg];
    std::memset(infile_mean, 0, sizeof(float) * nimg);
    std::memset(counts, 0, sizeof(float) * nimg);
    cudaStreamSynchronize(pimpl->dev.stream);
    for (int k = 0; k < nblocks; k++) {
      int id = k / ((l * l) / BLOCK_SIZE);
      infile_mean[id] += pimpl->host.reduction_buf[2 * k];
      counts[id] += pimpl->host.reduction_buf[2 * k + 1];
    }
    for (int k = 0; k < nimg; k++) {
      infile_mean[k] = std::sqrt(infile_mean[k] / (counts[k] * counts[k]));
    }

    // Do Normalization with computed infile_mean[]
    cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * nimg,
                    cudaMemcpyHostToDevice, pimpl->dev.stream);
    // Contain ap2ri
    normalize<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), l, l,
                                                             pimpl->dev.means.get());

    // Inplace IFT
    cufftExecC2C(pimpl->dev.fft.image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(),
                 CUFFT_INVERSE);
  }

  cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNoNorm::PickParticles(std::vector<float>& scores, float euler3) {
  int l = padding_size;
  int blockGPU_num = padded_template_size * batch_size / BLOCK_SIZE;
  int blockIMG_num = padded_template_size * block_x * block_y / BLOCK_SIZE;
  // rotate subIMG with angle "euler3"
  rotate_subIMG<<<blockIMG_num, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_image.get(), pimpl->dev.padded_rotated_image.get(), euler3, padding_size);

  // Inplace FFT
  cufftExecC2C(pimpl->dev.fft.image, pimpl->dev.padded_rotated_image.get(),
               pimpl->dev.padded_rotated_image.get(), CUFFT_FORWARD);

  // Scale IMG to normal size
  scale<<<blockIMG_num, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
      pimpl->dev.padded_rotated_image.get(), padded_template_size * block_x * block_y,
      padded_template_size);

  std::fill(scores.begin(), scores.end(), 0.f);
  // compute score for each block
  for (int j = 0; j < block_y; j++) {
    for (int i = 0; i < block_x; i++) {
      // compute CCG
      compute_corner_CCG<<<blockGPU_num, BLOCK_SIZE, 0, pimpl->dev.stream>>>(
          pimpl->dev.CCG.get(), pimpl->dev.padded_templates.get(),
          pimpl->dev.padded_rotated_image.get(), l, i + j * block_x);
      // Inplace IFT
      cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(),
                   CUFFT_INVERSE);
      // find peak(position) and get sum of data,data^2
      get_peak_and_SUM<<<blockGPU_num, BLOCK_SIZE, 4 * BLOCK_SIZE * sizeof(float),
                         pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.reduction_buf.get(),
                                              l, para.d_m);
      cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(),
                      4 * sizeof(float) * padded_template_size * batch_size / BLOCK_SIZE,
                      cudaMemcpyDeviceToHost, pimpl->dev.stream);

      std::memset(pimpl->host.ubuf.get(), 0, 4 * sizeof(float) * batch_size);
      // peak, sum of data[i], sum of data[i]^2
      float* peaks = pimpl->host.ubuf.get();
      float* pos = pimpl->host.ubuf.get() + batch_size;
      float* sums = pimpl->host.ubuf.get() + 2 * batch_size;
      float* sum2s = pimpl->host.ubuf.get() + 3 * batch_size;

      cudaStreamSynchronize(pimpl->dev.stream);

      // After Reduction -> compute mean for each image
      for (int k = 0; k < (padded_template_size * batch_size) / BLOCK_SIZE; k++) {
        int id = k / (padded_template_size / BLOCK_SIZE);
        if (peaks[id] < pimpl->host.reduction_buf[4 * k]) {
          peaks[id] = pimpl->host.reduction_buf[4 * k];
          pos[id] = pimpl->host.reduction_buf[4 * k + 1];
        }
        sums[id] += pimpl->host.reduction_buf[4 * k + 2];
        sum2s[id] += pimpl->host.reduction_buf[4 * k + 3];
      }

      // Update global score with local-block score for each template
      for (int k = 0; k < batch_size; k++) {
        float ra = sums[k] - peaks[k];
        float rb = sum2s[k] - peaks[k] * peaks[k];
        float rc = padded_template_size - 1;
        float sd = std::sqrt(rb / rc - (ra / rc) * (ra / rc));
        float score;
        if (sd == 0)
          score = 0;
        else
          score = peaks[k] / std::sqrt(rb / rc - (ra / rc) * (ra / rc));

        int cx = (int)pos[k] % l;
        int cy = (int)pos[k] / l;
        float centerx = i * (l - overlap) + (cx - l / 2) * cos(euler3 * PI / 180) +
                        (cy - l / 2) * sin(euler3 * PI / 180) + l / 2;  // centerx
        float centery = j * (l - overlap) + (cy - l / 2) * cos(euler3 * PI / 180) -
                        (cx - l / 2) * sin(euler3 * PI / 180) + l / 2;  // centery

        if (scores[3 * k] < score) {
          if (centerx >= 0 && centerx < nx && centery >= 0 && centery < ny) {
            scores[3 * k] = score;
            scores[3 * k + 1] = centerx;
            scores[3 * k + 2] = centery;
          }
        }
      }
    }
  }
}

void SearchNoNorm::OutputScore(std::string & output, float euler3, std::vector<float>& scores, const TileImages& tiles, const TileImages::Tile& tile) 
{
  char buf[1024];
  std::fstream out(output, std::ios::out | std::ios::trunc);
  for (int i = 0; i < batch_size; i++) {
    float score = scores[3 * i];
    float centerx = scores[3 * i + 1];
    float centery = scores[3 * i + 2];

    if (score >= para.thresh) {
      std::snprintf(
          buf, 1024,
          "%d\t%s\tdefocus=%f\tdfdiff=%f\tdfang=%f\teuler=%f,%f,%f\tcenter=%f,%f\tscore=%f\n",
          tiles.unused, tiles.rpath.c_str(), -para.defocus, para.dfdiff, para.dfang,
          euler.euler1[i], euler.euler2[i], euler3, centerx, centery, score);
      out << buf;
      ++line_count;
    }
  }
}

void SearchNoNorm::work(const Templates& temp, const TileImages& tiles, std::string & output) 
{
  int idx = 1;

  std::printf("Device %d: Load template\n", pimpl->dev.id);
  LoadTemplate(temp);

  std::vector<float> scores(batch_size * 3);
  auto params = tiles.p;
  SetParams(params);
  std::printf("Device %d: Preprocess template\n", pimpl->dev.id);
  PreprocessTemplate();
  for (const auto& tile : tiles) {
    std::printf("Device %d: Tile %d: center (%d, %d)\n", pimpl->dev.id, idx++, tile.center.x,
                tile.center.y);
    std::printf("Device %d: Load image\n", pimpl->dev.id);
    LoadImage(tile);
    std::printf("Device %d: Preprocess image\n", pimpl->dev.id);
    PreprocessImage();
    std::printf("Device %d: Split image\n", pimpl->dev.id);
    SplitImage();
    std::printf("Device %d: Rotate and pick, euler3 in [0, 360), step = %f\n", pimpl->dev.id,
                phi_step);
    for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) {
      PickParticles(scores, euler3);
      OutputScore(output, euler3, scores, tiles, tile);
    };
    std::printf("\n");
    std::printf("Device %d: Current output line count: %d\n", pimpl->dev.id, line_count);
  }
}

void SearchNoNorm::work_verbose(const Templates & temp, const TileImages & tiles, std::string & output) 
{
  int idx = 1;

  std::printf("Device %d: Load template\n", pimpl->dev.id);
  LoadTemplate(temp);

  std::vector<float> scores(batch_size * 3);
  auto params = tiles.p;
  SetParams(params);
  std::printf("Device %d: Preprocess template\n", pimpl->dev.id);
  PreprocessTemplate();
  for (const auto& tile : tiles) 
  {
    std::printf("Device %d: Tile %d: center (%d, %d)\n", pimpl->dev.id, idx++, tile.center.x, tile.center.y);
    std::printf("Device %d: Load image\n", pimpl->dev.id);
    LoadImage(tile);
    std::printf("Device %d: Preprocess image\n", pimpl->dev.id);
    PreprocessImage();
    std::printf("Device %d: Split image\n", pimpl->dev.id);
    SplitImage();
    std::printf("Device %d: Rotate and pick, euler3 in [0, 360), step = %f\n", pimpl->dev.id, phi_step);
    for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) 
    {
      PickParticles(scores, euler3);
      OutputScore(output, euler3, scores, tiles, tile);
      if (static_cast<int>(euler3) % 10 == 0) 
      {
        std::printf(".");
        std::fflush(stdout);
      }
    };
    std::printf("\n");
    std::printf("Device %d: Current output line count: %d\n", pimpl->dev.id, line_count);
  }
}