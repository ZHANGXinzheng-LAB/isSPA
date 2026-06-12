#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <fstream>

#include "fft.cuh"
#include "helper.cuh"
#include "kernels.cuh"
#include "norm.cuh"
#include "smartptr.cuh"
#include "mrc_writer.h"

struct SearchNorm::impl 
{
  struct Device 
  {
    struct FFT 
    {
      cufftHandle templates;
      cufftHandle image;
      cufftHandle raw_image;
    } fft;

    cudaStream_t stream;

    device_unique_ptr<cufftComplex[]> padded_templates;
    //device_unique_ptr<float[]> padded_templates1;
    //device_unique_ptr<float[]> padded_templates2;
    device_unique_ptr<cufftComplex[]> CCG;
    device_unique_ptr<cufftComplex[]> CCG_sum;
    device_unique_ptr<cufftComplex[]> CCG_buf;

    device_unique_ptr<cufftComplex[]> padded_image;

    device_unique_ptr<float[]> image;
    device_unique_ptr<float[]> ra;
    device_unique_ptr<float[]> rb;
    device_unique_ptr<float[]> reduction_buf;
    device_unique_ptr<float[]> reduction_buf1;
    device_unique_ptr<float[]> means;

    int id;
  } dev;

  struct Host 
  {
    pinned_unique_ptr<float[]> reduction_buf;
    pinned_unique_ptr<float[]> reduction_buf1;
    std::unique_ptr<float[]> ubuf;
  } host;
};

SearchNorm::SearchNorm(const Config & c, const EulerData & e, Size img, int device)
    : para(c),
      euler(e),
      padding_size(c.geti("Window_size")),
      overlap(c.geti("Overlap")),
      invert(c.geti("Invert")),
      phase_flip(c.geti("Phase_flip")),
      phi_step(c.getf("Phi_step")),
      batch_size(e.size()),
      bin(c.geti("Bin")),
      nx(img.width),
      ny(img.height),
      line_count(0),
      pimpl(std::make_unique<impl>()) 
{
    std::cout << "Thread " << std::this_thread::get_id() << " sets device " << device << std::endl;
    pimpl->dev.id = device;
    cudaSetDevice(device);

    padded_template_size = padding_size * padding_size; // e.g. 512*512
    image_size = nx * ny; 

    auto blocks_one_axis = [](int length, int padding, int overlap) 
    {
        // 计算每个搜索框的起始位置并记录到阵列中
        std::vector<int> block_offsets{0,};
        int offset = 0;
        while (offset + padding < length) 
        {
            offset += padding - overlap;
            if (offset + padding >= length) 
            {
                offset = length - padding;
            }
            block_offsets.emplace_back(offset);
        }
        return block_offsets;
    };
    // num of blocks in x, y axis
    block_offsets_x = blocks_one_axis(nx, padding_size, overlap); // 每个搜索框的起点位置
    block_offsets_y = blocks_one_axis(ny, padding_size, overlap);
    block_x = block_offsets_x.size(); // 搜索框个数
    block_y = block_offsets_y.size();
    std::printf("Split the image with %d x_blocks and %d y_blocks\n", block_x, block_y);

    N_pixel = padded_template_size * batch_size;
    grid_size = (N_pixel - 1) / BLOCK_SIZE + 1;
    nimg = block_x * block_y;
    N_pixel1 = nimg * padded_template_size;
    grid_size1 = (N_pixel1 - 1) / BLOCK_SIZE + 1;
    //std::cout << grid_size <<std::endl;
    //std::cout << grid_size1 <<std::endl;

    //pimpl->dev.padded_templates1 = make_device_unique<float[]>(N_pixel);
    //pimpl->dev.padded_templates2 = make_device_unique<float[]>(N_pixel);
    pimpl->dev.padded_templates = make_device_unique<cufftComplex[]>(N_pixel);
    pimpl->dev.CCG = make_device_unique<cufftComplex[]>(N_pixel);
    pimpl->dev.CCG_sum = make_device_unique<cufftComplex[]>(N_pixel1);
    pimpl->dev.CCG_buf = make_device_unique<cufftComplex[]>(N_pixel);
    pimpl->dev.ra = make_device_unique<float[]>(batch_size * RA_SIZE);
    pimpl->dev.rb = make_device_unique<float[]>(batch_size * RA_SIZE);
    auto buf_size = 2 * grid_size;
    pimpl->host.reduction_buf = make_host_unique_pinned<float[]>(buf_size);
    pimpl->dev.reduction_buf = make_device_unique<float[]>(buf_size);
    auto buf_size1 = 2 * grid_size1;
    pimpl->host.reduction_buf1 = make_host_unique_pinned<float[]>(buf_size1);
    pimpl->dev.reduction_buf1 = make_device_unique<float[]>(buf_size1);

    pimpl->dev.means = make_device_unique<float[]>(batch_size);
    pimpl->host.ubuf = std::make_unique<float[]>(2 * batch_size);

    pimpl->dev.fft.templates = MakeFFTPlan(padding_size, padding_size, batch_size);
    cufftSetStream(pimpl->dev.fft.templates, pimpl->dev.stream);

    pimpl->dev.image = make_device_unique<float[]>(image_size);
    pimpl->dev.padded_image = make_device_unique<cufftComplex[]>(N_pixel1);

    pimpl->dev.fft.image = MakeFFTPlan(padding_size, padding_size, nimg);
    cufftSetStream(pimpl->dev.fft.image, pimpl->dev.stream);
    pimpl->dev.fft.raw_image = MakeFFTPlan(ny, nx, 1);
    cufftSetStream(pimpl->dev.fft.raw_image, pimpl->dev.stream);

    cudaStreamSynchronize(pimpl->dev.stream);

    cudaStreamCreate(&pimpl->dev.stream);
    CHECK();
    DeviceMemoryUsage();
}

SearchNorm::~SearchNorm() 
{
    cufftDestroy(pimpl->dev.fft.raw_image);
    cufftDestroy(pimpl->dev.fft.image);
    cufftDestroy(pimpl->dev.fft.templates);
    cudaStreamDestroy(pimpl->dev.stream);
}

void SearchNorm::LoadTemplate(const Templates & temp) 
{
    auto padded_templates = std::make_unique<cufftComplex[]>(N_pixel);
    std::memset(padded_templates.get(), 0, sizeof(cufftComplex) * N_pixel);

    // padding
    int sx = (padding_size - temp.width) / 2; // 两边都补零
    int sy = (padding_size - temp.height) / 2;
    const size_t size = temp.width * temp.height;
// 并行
#pragma omp parallel for
  for (int n = 0; n < batch_size; ++n) 
  {
    for (int j = 0; j < temp.height; j++) 
    {
      for (int i = 0; i < temp.width; i++) 
      {
        size_t index = padded_template_size * n + (sy + j) * padding_size + (sx + i); // 像素点指标
        float cur = temp.data[n * size + j * temp.width + i]; //像素点的值
        padded_templates[index].x = cur;
      }
    }
  }
    cudaMemcpyAsync(pimpl->dev.padded_templates.get(), padded_templates.get(), sizeof(cufftComplex) * N_pixel, cudaMemcpyHostToDevice, pimpl->dev.stream);
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::LoadImage(const Image & img) 
{
    double sum = 0, sum_s2 = 0;
    for (int i = 0; i < image_size; i++) 
    {
        double cur = img.data[i];
        sum += cur;
        sum_s2 += (cur * cur); //平方和
    }

    float avg = static_cast<float>(sum) / image_size; // 平均值
    float std = sqrt(static_cast<float>(sum_s2)  / (image_size) - avg * avg); // 标准差
    int up_bound = avg + 6 * std; // 以6倍标准差为上界
    int low_bound = avg - 6 * std; // 以6倍标准差为下界
    if (std > 0) 
    {
        #pragma omp parallel for
        for (int i = 0; i < image_size; i++)
            if (img.data[i] > up_bound || img.data[i] < low_bound) img.data[i] = avg; // 将超过上下界的像素值设为平均值
    }
    if (invert) 
    {
        #pragma omp parallel for
        for (int i = 0; i < nx * ny; i++) img.data[i] = -img.data[i];
    }

    cudaMemcpyAsync(pimpl->dev.image.get(), img.data.get(), sizeof(float) * image_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::SetParams(const Image::Params & params) 
{
    // set params
    para.defocus = params.defocus;
    para.dfang = params.dfang - 90;
    para.dfdiff = params.dfdiff;
    para.dfu = params.defocus + params.dfdiff;  // -defocus is minus, so abs(dfu) < abs(dfv)
    para.dfv = params.defocus - params.dfdiff;
    para.lambda = 12.2643 / sqrt(para.energy * 1000.0 + 0.97848 * para.energy * para.energy); // 利用相对论公式计算，波长以埃为单位
    para.ds = 1 / (para.apix * padding_size); // 照片边长的倒数，以埃为单位
}

void SearchNorm::PreprocessTemplate() 
{
    cudaMemsetAsync(pimpl->dev.CCG.get(), 0, sizeof(cufftComplex) * N_pixel, pimpl->dev.stream);

    cudaMemsetAsync(pimpl->dev.reduction_buf.get(), 0, 2 * sizeof(float) * grid_size, pimpl->dev.stream);
    /*
    // generate Mask, count number of non-zero digits
    generate_mask<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(l, pimpl->dev.CCG.get(), r, pimpl->dev.reduction_buf.get(), up_bound, low_bound);

    cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), sizeof(float) * nblocks, cudaMemcpyDeviceToHost, pimpl->dev.stream);

    std::memset(pimpl->host.ubuf.get(), 0, batch_size * sizeof(float));

    cudaStreamSynchronize(pimpl->dev.stream);
    for (int k = 0; k < nblocks; k++) 
    {
        int id = k * BLOCK_SIZE / padded_template_size;
        pimpl->host.ubuf[id] += pimpl->host.reduction_buf[k];
    }

    // Calculate dot of mask and all templates
    cudaMemcpyAsync(pimpl->dev.means.get(), pimpl->host.ubuf.get(), sizeof(float) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);

    multiCount_dot<<<nblocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(l, pimpl->dev.CCG.get(), pimpl->dev.padded_templates.get(), pimpl->dev.means.get(), pimpl->dev.reduction_buf.get());

    cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), sizeof(float) * nblocks, cudaMemcpyDeviceToHost, pimpl->dev.stream);

    std::memset(pimpl->host.ubuf.get(), 0, batch_size * sizeof(float));

    cudaStreamSynchronize(pimpl->dev.stream);

    for (int k = 0; k < nblocks; k++) 
    {
        int id = k * BLOCK_SIZE / padded_template_size;
        pimpl->host.ubuf[id] += pimpl->host.reduction_buf[k];
    }

    // put em on GPU
    cudaMemcpyAsync(pimpl->dev.means.get(), pimpl->host.ubuf.get(), sizeof(float) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
    */
    // 对模板求和、求平方和
    UpdateSigma<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float) * 2, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.reduction_buf.get(), N_pixel);

    cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), 2 * sizeof(float) * grid_size, cudaMemcpyDeviceToHost, pimpl->dev.stream);

    std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));

    cudaStreamSynchronize(pimpl->dev.stream);
    for (int k = 0; k < grid_size; k++) 
    {
        // int id = k * BLOCK_SIZE / padded_template_size;
        pimpl->host.ubuf[2 * (k * BLOCK_SIZE / padded_template_size)] += pimpl->host.reduction_buf[2 * k];
        // sum of value
        pimpl->host.ubuf[2 * (k * BLOCK_SIZE / padded_template_size) + 1] += pimpl->host.reduction_buf[2 * k + 1];
        // sum of value^2
    }

    auto sigmas = make_host_unique_pinned<double[]>(batch_size);
    for (int i = 0; i < batch_size; i++) 
    {
        double mean = pimpl->host.ubuf[2 * i] / (double)(image_size); // 计算每张模板的平均值
        sigmas[i] = std::sqrt(pimpl->host.ubuf[2 * i + 1] / (double)(image_size) - mean * mean); // 计算每张模板照片的标准差
        if (sigmas[i] < 0 || !std::isfinite(sigmas[i])) sigmas[i] = 0;
    }

    auto dev_sigmas = make_device_unique<double[]>(batch_size);
    // data[i]=(data[i]-em)/s;
    cudaMemcpyAsync(dev_sigmas.get(), sigmas.get(), sizeof(double) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
    scale_each<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(padding_size, pimpl->dev.padded_templates.get(), pimpl->dev.means.get(), dev_sigmas.get(), N_pixel);

    // **************************************************************
    // apply whitening filter and do ift
    // input: Padded IMAGE (Real SPACE)
    // output: IMAGE_whiten (Fourier SPACE in RI)
    // **************************************************************
    cudaMemsetAsync(pimpl->dev.ra.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);
    cudaMemsetAsync(pimpl->dev.rb.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);

    // Inplace FFT 傅里叶变换
    cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(), pimpl->dev.padded_templates.get(), CUFFT_FORWARD);

    // CUFFT will enlarge VALUE to N times. Restore it
    scale<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), N_pixel, padded_template_size);

    // Whiten at fourier space
    // contain ri2ap
    SQRSum_by_circle<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size, padding_size, N_pixel);

    // contain ap2ri
    whiten_Tmp<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size, N_pixel);

    // **************************************************************
    // 1. lowpass
    // 2. apply weighting function
    // 3. normlize
    // input: masked_whiten_IMAGE (Fourier SPACE in RI)
    // output: PROCESSED_IMAGE (Fourier SPACE in AP)
    // **************************************************************

    //auto padded_templates1 = std::make_unique<float[]>(N_pixel);
    //auto padded_templates2 = std::make_unique<float[]>(N_pixel);

    apply_weighting_function<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), padding_size, para, N_pixel);
    //cudaMemcpyAsync(padded_templates1.get(), pimpl->dev.padded_templates1.get(), sizeof(float) * N_pixel, cudaMemcpyDeviceToHost, pimpl->dev.stream);
    //cudaMemcpyAsync(padded_templates2.get(), pimpl->dev.padded_templates2.get(), sizeof(float) * N_pixel, cudaMemcpyDeviceToHost, pimpl->dev.stream);
    //cudaStreamSynchronize(pimpl->dev.stream);
    //saveDataToFile(padded_templates1.get(), N_pixel, "/cyh/inter_files/isSPA/isSPA_test/ctf0_old.txt");
    //saveDataToFile(padded_templates2.get(), N_pixel, "/cyh/inter_files/isSPA/isSPA_test/ctf1_new.txt");

    compute_area_sum_ofSQR<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.reduction_buf.get(), padding_size, N_pixel);

    cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), 2 * sizeof(float) * grid_size, cudaMemcpyDeviceToHost, pimpl->dev.stream);

    // After Reduction -> compute mean for each image
    std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));
    float * infile_mean = pimpl->host.ubuf.get();
    float * counts = pimpl->host.ubuf.get() + batch_size;

    cudaStreamSynchronize(pimpl->dev.stream);
    for (int k = 0; k < grid_size; k++) 
    {
        //int id = k * BLOCK_SIZE / padded_template_size;
        infile_mean[k * BLOCK_SIZE / padded_template_size] += pimpl->host.reduction_buf[2 * k]; // 对每张照片内的点模方求和
        counts[k * BLOCK_SIZE / padded_template_size] += pimpl->host.reduction_buf[2 * k + 1];
    }

    for (int k = 0; k < batch_size; k++) 
    {
        infile_mean[k] = std::sqrt(infile_mean[k] / (counts[k] * counts[k]));
    }
    // Do Normalization with computed infile_mean[]
    cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
    // Contain ap2ri
    normalize<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), padded_template_size, pimpl->dev.means.get(), N_pixel);
    cudaMemsetAsync(pimpl->dev.means.get(), 0, batch_size * sizeof(float), pimpl->dev.stream);
    // 傅里叶逆变换
    cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(), pimpl->dev.padded_templates.get(), CUFFT_INVERSE);

    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::PreprocessTemplate(const std::vector<float> & k, const std::vector<float> & fsc) 
{
    cudaMemsetAsync(pimpl->dev.CCG.get(), 0, sizeof(cufftComplex) * N_pixel, pimpl->dev.stream);

    cudaMemsetAsync(pimpl->dev.reduction_buf.get(), 0, 2 * sizeof(float) * grid_size, pimpl->dev.stream);

    // 对模板求和、求平方和
    UpdateSigma<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float) * 2, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.reduction_buf.get(), N_pixel);

    cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), 2 * sizeof(float) * grid_size, cudaMemcpyDeviceToHost, pimpl->dev.stream);

    std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));

    cudaStreamSynchronize(pimpl->dev.stream);
    for (int k = 0; k < grid_size; k++) 
    {
        // int id = k * BLOCK_SIZE / padded_template_size;
        pimpl->host.ubuf[2 * (k * BLOCK_SIZE / padded_template_size)] += pimpl->host.reduction_buf[2 * k];
        // sum of value
        pimpl->host.ubuf[2 * (k * BLOCK_SIZE / padded_template_size) + 1] += pimpl->host.reduction_buf[2 * k + 1];
        // sum of value^2
    }

    auto sigmas = make_host_unique_pinned<double[]>(batch_size);
    for (int i = 0; i < batch_size; i++) 
    {
        double mean = pimpl->host.ubuf[2 * i] / (double)(image_size); // 计算每张模板的平均值
        sigmas[i] = std::sqrt(pimpl->host.ubuf[2 * i + 1] / (double)(image_size) - mean * mean); // 计算每张模板照片的标准差
        if (sigmas[i] < 0 || !std::isfinite(sigmas[i])) sigmas[i] = 0;
    }

    auto dev_sigmas = make_device_unique<double[]>(batch_size);
    // data[i]=(data[i]-em)/s;
    cudaMemcpyAsync(dev_sigmas.get(), sigmas.get(), sizeof(double) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
    scale_each<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(padding_size, pimpl->dev.padded_templates.get(), pimpl->dev.means.get(), dev_sigmas.get(), N_pixel);

    // **************************************************************
    // apply whitening filter and do ift
    // input: Padded IMAGE (Real SPACE)
    // output: IMAGE_whiten (Fourier SPACE in RI)
    // **************************************************************
    cudaMemsetAsync(pimpl->dev.ra.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);
    cudaMemsetAsync(pimpl->dev.rb.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);

    // Inplace FFT 傅里叶变换
    cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(), pimpl->dev.padded_templates.get(), CUFFT_FORWARD);

    // CUFFT will enlarge VALUE to N times. Restore it
    scale<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), N_pixel, padded_template_size);

    // Whiten at fourier space
    // contain ri2ap
    SQRSum_by_circle<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size, padding_size, N_pixel);

    // contain ap2ri
    whiten_Tmp<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size, N_pixel);

    // **************************************************************
    // 1. lowpass
    // 2. apply weighting function
    // 3. normlize
    // input: masked_whiten_IMAGE (Fourier SPACE in RI)
    // output: PROCESSED_IMAGE (Fourier SPACE in AP)
    // **************************************************************

    //auto padded_templates1 = std::make_unique<float[]>(N_pixel);
    //std::memset(padded_templates1.get(), 0, sizeof(float) * N_pixel);
    //auto padded_templates2 = std::make_unique<float[]>(N_pixel);
    float * d_k, * d_fsc;
    int size_fsc = fsc.size();
    cudaMalloc(&d_k, sizeof(float) * size_fsc);
    cudaMalloc(&d_fsc, sizeof(float) * size_fsc);
    cudaMemcpyAsync(d_k, k.data(), sizeof(float) * size_fsc, cudaMemcpyHostToDevice, pimpl->dev.stream);
    cudaMemcpyAsync(d_fsc, fsc.data(), sizeof(float) * size_fsc, cudaMemcpyHostToDevice, pimpl->dev.stream);

    apply_weighting_function<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), padding_size, para, N_pixel, d_k, d_fsc, size_fsc-1);

    //cudaMemcpyAsync(padded_templates1.get(), pimpl->dev.padded_templates1.get(), sizeof(float) * N_pixel, cudaMemcpyDeviceToHost, pimpl->dev.stream);
    //cudaMemcpyAsync(padded_templates2.get(), pimpl->dev.padded_templates2.get(), sizeof(float) * N_pixel, cudaMemcpyDeviceToHost, pimpl->dev.stream);
    //cudaStreamSynchronize(pimpl->dev.stream);
    //write_mrc("fsc.mrc", padded_templates1.get(), padding_size, padding_size, batch_size, 1.36f);
    //saveDataToFile(padded_templates1.get(), N_pixel, "/cyh/inter_files/isSPA/ctf0_old.txt");
    //saveDataToFile(padded_templates2.get(), N_pixel, "/cyh/inter_files/isSPA/isSPA_test/ctf1_new.txt");

    compute_area_sum_ofSQR<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.reduction_buf.get(), padding_size, N_pixel);

    cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), 2 * sizeof(float) * grid_size, cudaMemcpyDeviceToHost, pimpl->dev.stream);

    // After Reduction -> compute mean for each image
    std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));
    float * infile_mean = pimpl->host.ubuf.get();
    float * counts = pimpl->host.ubuf.get() + batch_size;

    cudaStreamSynchronize(pimpl->dev.stream);
    for (int k = 0; k < grid_size; k++) 
    {
        //int id = k * BLOCK_SIZE / padded_template_size;
        infile_mean[k * BLOCK_SIZE / padded_template_size] += pimpl->host.reduction_buf[2 * k]; // 对每张照片内的点模方求和
        counts[k * BLOCK_SIZE / padded_template_size] += pimpl->host.reduction_buf[2 * k + 1];
    }

    for (int k = 0; k < batch_size; k++) 
    {
        infile_mean[k] = std::sqrt(infile_mean[k] / (counts[k] * counts[k]));
    }
    // Do Normalization with computed infile_mean[]
    cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
    // Contain ap2ri
    normalize<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), padded_template_size, pimpl->dev.means.get(), N_pixel);
    cudaMemsetAsync(pimpl->dev.means.get(), 0, batch_size * sizeof(float), pimpl->dev.stream);
    // 傅里叶逆变换
    cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(), pimpl->dev.padded_templates.get(), CUFFT_INVERSE);

    cudaStreamSynchronize(pimpl->dev.stream);
    cudaFree(d_k);
    cudaFree(d_fsc);
}

void SearchNorm::PreprocessImage(const Image & img) 
{
    int nblocks = (image_size - 1) / BLOCK_SIZE + 1;

    cudaMemsetAsync(pimpl->dev.padded_image.get(), 0, N_pixel1 * sizeof(cufftComplex), pimpl->dev.stream);

    float2Complex<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.image.get(), image_size);
    // fft inplace
    cufftExecC2C(pimpl->dev.fft.raw_image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(), CUFFT_FORWARD);
    scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), image_size, image_size);

    // phase flipping
    //do_phase_flip<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), para, nx, ny);

    // Whiten at fourier space
    cudaMemsetAsync(pimpl->dev.ra.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);
    cudaMemsetAsync(pimpl->dev.rb.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);

    // contain ri2ap
    SQRSum_by_circle<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, image_size);

    // 1. whiten
    // 2. low pass
    // 3. weight
    // 4. ap2ri

    whiten_filter_weight_Img<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, para);

    // 0Hz -> 0
    //set_0Hz_to_0_at_RI<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get());

    // ifft inplace
    cufftExecC2C(pimpl->dev.fft.raw_image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(), CUFFT_INVERSE);
    Complex2float<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.image.get(), pimpl->dev.padded_image.get(), image_size);

    cudaStreamSynchronize(pimpl->dev.stream);
    pimpl->dev.ra = nullptr;
    pimpl->dev.rb = nullptr;
}

void SearchNorm::SplitImage() 
{
    cudaMemsetAsync(pimpl->dev.padded_image.get(), 0, N_pixel1 * sizeof(cufftComplex), pimpl->dev.stream);

    auto d_off_x = make_device_unique<int[]>(block_x);
    auto d_off_y = make_device_unique<int[]>(block_y);
    cudaMemcpyAsync(d_off_x.get(), block_offsets_x.data(), sizeof(int) * block_x, cudaMemcpyHostToDevice, pimpl->dev.stream);
    cudaMemcpyAsync(d_off_y.get(), block_offsets_y.data(), sizeof(int) * block_y, cudaMemcpyHostToDevice, pimpl->dev.stream);

    // split Image into blocks with overlap
    split_IMG<<<grid_size1, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.image.get(), pimpl->dev.padded_image.get(), d_off_x.get(), d_off_y.get(), nx, ny, padding_size, block_x, N_pixel1);

    cudaStreamSynchronize(pimpl->dev.stream);
    pimpl->dev.image = nullptr;

    // do normalize to all subIMGs
    // Inplace FFT
    cufftExecC2C(pimpl->dev.fft.image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(), CUFFT_FORWARD);
    // Scale IMG to normal size
    scale<<<grid_size1, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), N_pixel1, padded_template_size);
    ri2ap<<<grid_size1, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), N_pixel1);
    compute_area_sum_ofSQR<<<grid_size1, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.reduction_buf1.get(), padding_size, N_pixel1);

    cudaMemcpyAsync(pimpl->host.reduction_buf1.get(), pimpl->dev.reduction_buf1.get(), 2 * grid_size1 * sizeof(float), cudaMemcpyDeviceToHost, pimpl->dev.stream);

    // After Reduction -> compute mean for each image
    float infile_mean[nimg], counts[nimg];
    std::memset(infile_mean, 0, sizeof(float) * nimg);
    std::memset(counts, 0, sizeof(float) * nimg);
    cudaStreamSynchronize(pimpl->dev.stream);
    for (int k = 0; k < grid_size1; k++) 
    {
        infile_mean[k * BLOCK_SIZE / padded_template_size] += pimpl->host.reduction_buf1[2 * k]; 
        counts[k * BLOCK_SIZE / padded_template_size] += pimpl->host.reduction_buf1[2 * k + 1];
        // std::cout << k << " " << counts[id] << " " << infile_mean[id] << std::endl;
    }
    for (int k = 0; k < nimg; k++) 
    {
        infile_mean[k] = std::sqrt(infile_mean[k] / (counts[k] * counts[k]));
    }

    // Do Normalization with computed infile_mean[]
    cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * nimg, cudaMemcpyHostToDevice, pimpl->dev.stream);
    // Contain ap2ri
    normalize<<<grid_size1, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), padded_template_size, pimpl->dev.means.get(), N_pixel1);

    cudaMemsetAsync(pimpl->dev.CCG_sum.get(), 0, sizeof(cufftComplex) * N_pixel1, pimpl->dev.stream);
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::RotateTemplate(float euler3) 
{
    // 旋转所有模板
    rotate_subIMG<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.CCG_buf.get(), euler3, padding_size, N_pixel);
    // 施加mask，将边缘点设为0，soft edge mask
    apply_mask<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), para.d_m, para.edge_width, padding_size, N_pixel);
    // 求和，求平方和
    compute_sum_sqr<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), pimpl->dev.reduction_buf.get(), N_pixel);
    cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), 2 * sizeof(float) * grid_size, cudaMemcpyDeviceToHost, pimpl->dev.stream);

    std::memset(pimpl->host.ubuf.get(), 0, 2 * batch_size * sizeof(float));
    float * infile_mean = pimpl->host.ubuf.get();
    float * infile_sqr = pimpl->host.ubuf.get() + batch_size;

    cudaStreamSynchronize(pimpl->dev.stream);
    for (int k = 0; k < grid_size; k++) 
    {
        //int id = k * BLOCK_SIZE / padded_template_size;
        infile_mean[k * BLOCK_SIZE / padded_template_size] += pimpl->host.reduction_buf[2 * k];
        infile_sqr[k * BLOCK_SIZE / padded_template_size] += pimpl->host.reduction_buf[2 * k + 1];
    }

    for (int k = 0; k < batch_size; k++) 
    {
        infile_mean[k] = infile_mean[k] / padded_template_size;
        infile_sqr[k] = infile_sqr[k] / padded_template_size - infile_mean[k] * infile_mean[k];
    }
    cudaMemcpyAsync(pimpl->dev.means.get(), infile_mean, sizeof(float) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
    substract_by_mean<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), padded_template_size, pimpl->dev.means.get(), N_pixel);
    cudaMemcpyAsync(pimpl->dev.means.get(), infile_sqr, sizeof(float) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
    divided_by_var<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), padded_template_size, pimpl->dev.means.get(), N_pixel);
    cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG_buf.get(), pimpl->dev.CCG_buf.get(), CUFFT_FORWARD);
    scale<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), N_pixel, padded_template_size);
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::ComputeCCGSum() 
{
    // compute score for each block
    for (int j = 0; j < block_y; ++j) 
    {
        for (int i = 0; i < block_x; ++i) 
        {
            // auto block_id = i + j * block_x;
            // compute CCG
            compute_corner_CCG<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.CCG_buf.get(), pimpl->dev.padded_image.get(), padding_size, i + j * block_x, N_pixel);
            // Inplace IFT
            cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(), CUFFT_INVERSE);
            // compute avg/variance
            add_CCG_to_sum<<<(padded_template_size - 1)/ BLOCK_SIZE + 1, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_sum.get(), pimpl->dev.CCG.get(), padded_template_size, batch_size, i + j * block_x);
        }
    }
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::ComputeCCGMean() 
{
    set_CCG_mean<<<grid_size1, BLOCK_SIZE>>>(pimpl->dev.CCG_sum.get(), N_pixel1, batch_size * (360 / phi_step));
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::PickParticles(std::vector<float>& scores, float euler3) 
{
    scores.clear();

    // compute score for each block
    for (int j = 0; j < block_y; j++) 
    {
        for (int i = 0; i < block_x; i++) 
        {
            // compute CCG
            compute_corner_CCG<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.CCG_buf.get(), pimpl->dev.padded_image.get(), padding_size, i + j * block_x, N_pixel);
            // Inplace IFT
            cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(), CUFFT_INVERSE);
            // update CCG with avg/var
            update_CCG<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_sum.get(), pimpl->dev.CCG.get(), padded_template_size, i + j * block_x, N_pixel);

            // find peak in each block
            get_peak_pos<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.reduction_buf.get(), padded_template_size, N_pixel);
            cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), 2 * sizeof(float) * grid_size, cudaMemcpyDeviceToHost, pimpl->dev.stream);
            cudaStreamSynchronize(pimpl->dev.stream);

            // After Reduction -> compute mean for each image
            for (int k = 0; k < grid_size; k++) 
            {
                if (pimpl->host.reduction_buf[2 * k] >= para.thresh) 
                {
                    float score = pimpl->host.reduction_buf[2 * k];
                    int centerx = block_offsets_x[i] + (int)pimpl->host.reduction_buf[2 * k + 1] % padding_size;
                    int centery = block_offsets_y[j] + (int)pimpl->host.reduction_buf[2 * k + 1] / padding_size;
                    if (centerx >= para.d_m / 2 && centerx < nx - para.d_m / 2 && centery >= para.d_m / 2 && centery < ny - para.d_m / 2) 
                    {
                        scores.emplace_back(score);
                        scores.emplace_back(centerx);
                        scores.emplace_back(centery);
                        scores.emplace_back(k * BLOCK_SIZE / padded_template_size);
                    }
                }
            }
        }
    }
}

void SearchNorm::OutputScore(std::string & output, std::vector<float> & scores, float euler3, const Image & input) 
{
    char buf[1024];
    std::fstream out(output, std::ios::out | std::ios::app);
    std::filesystem::path filePath = output;
    if (filePath.extension() == ".star")
    {
        euler3 = euler3 + 90;
        float df1 = -para.dfu * 10000;
        float df2 = -para.dfv * 10000;
        for (int i = 0; i < scores.size(); i += 4) 
        {
            float score = scores[i];
            float centerx = scores[i + 1];
            float centery = scores[i + 2];
            size_t j = scores[i + 3];
            std::snprintf(buf, 1024, "%s %f %f %f %f %f %f %f %f %d #%f\n", input.rpath.c_str(), centerx, centery, df1, df2, para.dfang, euler.euler2[j], euler.euler1[j], euler3, 1, score);
            out << buf;
            //std::cout << "Writing to the output file.\n";
            ++line_count;
        }
    }
    else
    {
        for (int i = 0; i < scores.size(); i += 4) 
        {
            float score = scores[i]; 
            float centerx = scores[i + 1]; 
            float centery = scores[i + 2]; 
            size_t j = scores[i + 3]; 
            std::snprintf(buf, 1024, "%d\t%s\tdefocus=%f\tdfdiff=%f\tdfang=%f\teuler=%f,%f,%f\tcenter=%f,%f\tscore=%f\n", input.unused, input.rpath.c_str(), para.defocus, para.dfdiff, para.dfang+90, euler.euler1[j], euler.euler2[j], euler3, centerx, centery, score); 
            out << buf; 
            //std::cout << "Writing to the output file.\n";
            ++line_count; 
        }
    }
}

void SearchNorm::work_verbose(const Templates & temp, const Image & image, std::string & output, const std::vector<float> & k, const std::vector<float> & fsc)
{
    using namespace std;
    vector<float> scores;
    auto params = image.p;

    SetParams(params);
    printf("Device %d: Loading templates\n", pimpl->dev.id);
    LoadTemplate(temp);

    printf("Device %d: Image: %s, (%zu, %zu)\n", pimpl->dev.id, image.rpath.c_str(), params.width, params.height);
    printf("Device %d: Loading image\n", pimpl->dev.id);
    LoadImage(image);

    printf("Device %d: Preprocessing templates\n", pimpl->dev.id);
    PreprocessTemplate(k, fsc);
    printf("Device %d: Preprocessing image\n", pimpl->dev.id);
    PreprocessImage(image);
    printf("Device %d: Spliting image\n", pimpl->dev.id);
    SplitImage();

    printf("Device %d: Computing the avgs and vars of all CCGs, euler3 in [0, 360), step = %f\n", pimpl->dev.id, phi_step);
    for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) 
    {
        RotateTemplate(euler3);
        ComputeCCGSum();
        if (static_cast<int>(euler3) % 72 == 0) 
        {
            printf(":) ");
            fflush(stdout);
        }
    }
    printf("\n");
    printf("Device %d: Computing CCG means\n", pimpl->dev.id);
    ComputeCCGMean();
    printf("Device %d: Updating CCGs and computing scores, euler3 in [0, 360), step = %f\n", pimpl->dev.id, phi_step);
    for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) 
    {
        RotateTemplate(euler3);
        PickParticles(scores, euler3);
        OutputScore(output, scores, euler3, image);
        if (static_cast<int>(euler3) % 72 == 0) 
        {
            printf(":) ");
            fflush(stdout);
        }
    };
    printf("\n");
    printf("Device %d: Current output line count: %d\n", pimpl->dev.id, line_count);
}

void SearchNorm::work_verbose(const Templates & temp, const Image & image, std::string & output) 
{
    using namespace std;
    vector<float> scores;
    auto params = image.p;

    SetParams(params);
    printf("Device %d: Loading templates\n", pimpl->dev.id);
    LoadTemplate(temp);

    printf("Device %d: Image: %s, (%zu, %zu)\n", pimpl->dev.id, image.rpath.c_str(), params.width, params.height);
    printf("Device %d: Loading image\n", pimpl->dev.id);
    LoadImage(image);

    printf("Device %d: Preprocessing templates\n", pimpl->dev.id);
    PreprocessTemplate();
    printf("Device %d: Preprocessing image\n", pimpl->dev.id);
    PreprocessImage(image);
    printf("Device %d: Spliting image\n", pimpl->dev.id);
    SplitImage();

    printf("Device %d: Computing the avgs and vars of all CCGs, euler3 in [0, 360), step = %f\n", pimpl->dev.id, phi_step);
    for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) 
    {
        RotateTemplate(euler3);
        ComputeCCGSum();
        if (static_cast<int>(euler3) % 72 == 0) 
        {
            printf(":) ");
            fflush(stdout);
        }
    }
    printf("\n");
    printf("Device %d: Computing CCG means\n", pimpl->dev.id);
    ComputeCCGMean();
    printf("Device %d: Updating CCGs and computing scores, euler3 in [0, 360), step = %f\n", pimpl->dev.id, phi_step);
    for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) 
    {
        RotateTemplate(euler3);
        PickParticles(scores, euler3);
        OutputScore(output, scores, euler3, image);
        if (static_cast<int>(euler3) % 72 == 0) 
        {
            printf(":) ");
            fflush(stdout);
        }
    };
    printf("\n");
    printf("Device %d: Current output line count: %d\n", pimpl->dev.id, line_count);
}

void SearchNorm::saveComplexToBinary(const cufftComplex* data, size_t size, const std::string& filename) 
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    // 直接写入二进制数据（假设 cuFFTComplex 是紧密排列的结构体）
    file.write(reinterpret_cast<const char*>(data), size * sizeof(cufftComplex));
    file.close();
}