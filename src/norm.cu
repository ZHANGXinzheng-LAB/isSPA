#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <fstream>
#include <cmath>
#include <string>

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
          cufftHandle templates1;
          cufftHandle image;
          cufftHandle raw_image;
        } fft; // 分别对应模板/照片的FFT变换

        cudaStream_t stream;

        device_unique_ptr<float[]> image; 
        device_unique_ptr<cufftComplex[]> padded_image; // 切割后的照片
        device_unique_ptr<cufftComplex[]> split_image;
        
        //cufftComplex * padded_templates;
        device_unique_ptr<cufftComplex[]> padded_templates; // 补零后的模板
        device_unique_ptr<cufftComplex[]> rotated_templates; // sub-pixel模板

        device_unique_ptr<cufftComplex[]> CCG; 
        device_unique_ptr<cufftComplex[]> CCG_sum; // CCG求和/平方求和
        device_unique_ptr<cufftComplex[]> CCG_buf; // 计算CCG前的模板

        

        // 存储中间变量
        device_unique_ptr<float[]> ra; 
        device_unique_ptr<float[]> rb;
        device_unique_ptr<float[]> reduction_buf;
        //device_unique_ptr<float[]> reduction_buf1;
        device_unique_ptr<float[]> reduction_buf2;
        device_unique_ptr<float[]> means; 

        device_unique_ptr<float[]> k; 
        device_unique_ptr<float[]> fsc;
        device_unique_ptr<float[]> k1;
        device_unique_ptr<float[]> n_k;

        device_unique_ptr<int[]> offset_x; // 照片切割位置的横纵坐标
        device_unique_ptr<int[]> offset_y;

        int id;
    } dev; // GPU中的变量

    struct Host 
    {
        pinned_unique_ptr<float[]> reduction_buf;
        //pinned_unique_ptr<float[]> reduction_buf1;
        pinned_unique_ptr<float[]> reduction_buf2;
        std::unique_ptr<float[]> ubuf;
    } host; //存储中间结果的变量
};

SearchNorm::SearchNorm(const Config & c, const EulerData & e, Size img, const TextFileData & text_data, int device, int fourier_pad)
    : para(c),
      euler(e),
      padding_size(c.geti("Window_size")),
      overlap(c.geti("Overlap")),
      invert(c.geti("Invert")),
      phi_step(c.getf("Phi_step")),
      batch_size(e.size()),
      nx(img.width),
      ny(img.height),
      total_line_count(0),
      pimpl(std::make_unique<impl>()) 
{
    std::cout << "Setting device " << device << std::endl;
    pimpl->dev.id = device;
    cudaSetDevice(device);

    if (fourier_pad == 0) ft_padding = padding_size;
    if (fourier_pad == 1) ft_padding = 768;
    if (fourier_pad == 2) ft_padding = 1024;
    padded_template_size = padding_size * padding_size; // e.g. 512*512
    ft_padded_size = ft_padding * ft_padding; // e.g. 768*768
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
    std::printf("Splitting the image with %d x_blocks and %d y_blocks\n", block_x, block_y);

    N_pixel = padded_template_size * batch_size;
    // std::cout << N_pixel << std::endl;
    grid_size = (N_pixel - 1) / BLOCK_SIZE + 1;
    nimg = block_x * block_y;
    N_pixel1 = nimg * padded_template_size;
    grid_size1 = (N_pixel1 - 1) / BLOCK_SIZE + 1;
    N_pixel2 = ft_padded_size * batch_size; // 补零后，模板的像素数
    grid_size2 = (N_pixel2 - 1) / BLOCK_SIZE + 1;
    N_pixel3 = ft_padded_size * nimg; // 补零后，照片的像素数
    grid_size3 = (N_pixel3 - 1) / BLOCK_SIZE + 1;

    pimpl->dev.offset_x = make_device_unique<int[]>(block_x);
    pimpl->dev.offset_y = make_device_unique<int[]>(block_y);

    //pimpl->dev.padded_templates2 = make_device_unique<float[]>(N_pixel);
    
    //cudaError_t err = cudaMallocManaged(&pimpl->dev.padded_templates, N_pixel*sizeof(cufftComplex));

    pimpl->dev.padded_templates = make_device_unique<cufftComplex[]>(N_pixel);

    if (ft_padding != padding_size)
    {
        pimpl->dev.rotated_templates = make_device_unique<cufftComplex[]>(N_pixel);
        pimpl->dev.split_image = make_device_unique<cufftComplex[]>(N_pixel3);
        pimpl->host.reduction_buf2 = make_host_unique_pinned<float[]>(2 * grid_size2);
        pimpl->dev.reduction_buf2 = make_device_unique<float[]>(grid_size2 * 2);
        pimpl->dev.fft.templates1 = MakeFFTPlan(ft_padding, ft_padding, batch_size);
        cufftSetStream(pimpl->dev.fft.templates1, pimpl->dev.stream);
    }
    pimpl->dev.CCG = make_device_unique<cufftComplex[]>(N_pixel2);
    pimpl->dev.CCG_sum = make_device_unique<cufftComplex[]>(N_pixel3);
    pimpl->dev.CCG_buf = make_device_unique<cufftComplex[]>(N_pixel2);

    pimpl->dev.image = make_device_unique<float[]>(image_size);
    pimpl->dev.padded_image = make_device_unique<cufftComplex[]>(N_pixel1);
    
    pimpl->dev.ra = make_device_unique<float[]>(batch_size * RA_SIZE);
    pimpl->dev.rb = make_device_unique<float[]>(batch_size * RA_SIZE);
    
    pimpl->host.reduction_buf = make_host_unique_pinned<float[]>(2 * grid_size);
    pimpl->dev.reduction_buf = make_device_unique<float[]>(2 * grid_size);
    //pimpl->host.reduction_buf1 = make_host_unique_pinned<float[]>(2 * grid_size1);
    //pimpl->dev.reduction_buf1 = make_device_unique<float[]>(2 * grid_size1);

    pimpl->dev.means = make_device_unique<float[]>(batch_size);
    pimpl->host.ubuf = std::make_unique<float[]>(2 * batch_size);

    pimpl->dev.fft.templates = MakeFFTPlan(padding_size, padding_size, batch_size);
    cufftSetStream(pimpl->dev.fft.templates, pimpl->dev.stream);

    pimpl->dev.fft.image = MakeFFTPlan(padding_size, padding_size, nimg);
    cufftSetStream(pimpl->dev.fft.image, pimpl->dev.stream);
    pimpl->dev.fft.raw_image = MakeFFTPlan(ny, nx, 1);
    cufftSetStream(pimpl->dev.fft.raw_image, pimpl->dev.stream);

    cudaStreamSynchronize(pimpl->dev.stream);

    cudaStreamCreate(&pimpl->dev.stream);

    cudaMemcpyAsync(pimpl->dev.offset_x.get(), block_offsets_x.data(), sizeof(int) * block_x, cudaMemcpyHostToDevice, pimpl->dev.stream);
    cudaMemcpyAsync(pimpl->dev.offset_y.get(), block_offsets_y.data(), sizeof(int) * block_y, cudaMemcpyHostToDevice, pimpl->dev.stream);

    if (!text_data.k.empty())
    {
        fsc_size = text_data.fsc.size();
        pimpl->dev.k = make_device_unique<float[]>(fsc_size);
        pimpl->dev.fsc = make_device_unique<float[]>(fsc_size);
        cudaMemcpyAsync(pimpl->dev.k.get(), text_data.k.data(), sizeof(float) * fsc_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
        cudaMemcpyAsync(pimpl->dev.fsc.get(), text_data.fsc.data(), sizeof(float) * fsc_size, cudaMemcpyHostToDevice, pimpl->dev.stream);    
    }
    if (!text_data.k1.empty())
    {
        nk_size = text_data.n_k.size();
        pimpl->dev.k1 = make_device_unique<float[]>(nk_size);
        pimpl->dev.n_k = make_device_unique<float[]>(nk_size);
        cudaMemcpyAsync(pimpl->dev.k1.get(), text_data.k1.data(), sizeof(float) * nk_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
        cudaMemcpyAsync(pimpl->dev.n_k.get(), text_data.n_k.data(), sizeof(float) * nk_size, cudaMemcpyHostToDevice, pimpl->dev.stream);    
    }

    //CHECK();
    DeviceMemoryUsage();
}

SearchNorm::~SearchNorm() 
{
    cufftDestroy(pimpl->dev.fft.raw_image);
    cufftDestroy(pimpl->dev.fft.image);
    cufftDestroy(pimpl->dev.fft.templates);
    if (pimpl->dev.fft.templates1)
    {
        cufftDestroy(pimpl->dev.fft.templates1);
    }
    cudaStreamDestroy(pimpl->dev.stream);
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

void SearchNorm::LoadTemplate(const Templates & temp) 
{
    //std::cout << "No. of elements: "<< N_pixel << std::endl;
    std::vector<cufftComplex> padded_templates(N_pixel);
    //auto padded_templates = std::make_unique<cufftComplex[]>(N_pixel);
    //std::memset(padded_templates.get(), 0, sizeof(cufftComplex) * N_pixel);

    // padding
    int sx = (padding_size - temp.width) / 2; // 两边都补零
    int sy = (padding_size - temp.height) / 2;
    const size_t size = temp.width * temp.height;

    //std::printf("Initializing templates\n");
    // 并行
    #pragma omp parallel for
    for (size_t n = 0; n < batch_size; ++n) 
    {
        for (int j = 0; j < temp.height; j++) 
        {
            for (int i = 0; i < temp.width; i++) 
            {
                size_t index = padded_template_size * n + (sy + j) * padding_size + (sx + i); // 像素点指标
                //std::cout << "Index: " << index << std::endl;
                if (index >= N_pixel) 
                {
                    #pragma omp critical
                    {
                        std::cerr << "错误: padded_templates越界! " 
                                  << "n=" << n << ", j=" << j << ", i=" << i
                                  << ", index=" << index << " >= N_pixel=" << N_pixel 
                                  << std::endl;
                    }
                    continue;
                }
                float cur = temp.data[n * size + j * temp.width + i]; //像素点的值
                padded_templates[index].x = cur;
            }
        }
    }
    
    //std::printf("Passed assignment\n");
    cudaMemcpyAsync(pimpl->dev.padded_templates.get(), padded_templates.data(), sizeof(cufftComplex) * N_pixel, cudaMemcpyHostToDevice, pimpl->dev.stream);

    cudaStreamSynchronize(pimpl->dev.stream);
    /*
    std::vector<cufftComplex> padded_templates;
    std::cout << "The maximum size of a vector: " <<  padded_templates.max_size() << std::endl;
    try
    {
        std::vector<cufftComplex> padded_templates(N_pixel);

        std::memset(padded_templates.data(), 0, sizeof(cufftComplex) * N_pixel);
        // padding
        int sx = (padding_size - temp.width) / 2; // 两边都补零
        int sy = (padding_size - temp.height) / 2;
        const size_t size = temp.width * temp.height;

        std::printf("Initializing templates\n");
    
    // 并行
    int sx = (padding_size - temp.width) / 2; // 两边都补零
    int sy = (padding_size - temp.height) / 2;
    const size_t size = temp.width * temp.height;

    //cudaMemsetAsync(pimpl->dev.padded_templates.get(), 0, N_pixel * sizeof(cufftComplex), pimpl->dev.stream);

    //cudaStreamSynchronize(pimpl->dev.stream);

    #pragma omp parallel for
    for (int n = 0; n < batch_size; ++n) 
    {
        for (int j = 0; j < temp.height; j++) 
        {
            for (int i = 0; i < temp.width; i++) 
            {
                size_t index = padded_template_size * n + (sy + j) * padding_size + (sx + i); // 像素点指标
                float cur = temp.data[n * size + j * temp.width + i]; //像素点的值
                pimpl->dev.padded_templates[index].x = cur;
            }
        }
    }
    
    for (int index = 0; index < N_pixel; index++)
    {
        int n = index / padded_template_size; // 照片编号
        int local_index = index % padded_template_size;
        int j = local_index / padding_size;
        int i = local_index % padding_size;
        // std::cout << "Image: " << n << ", y: " << j << ", x: " << i << ", sx: " << sx << ", sy: " << sy << std::endl;
        if (i < sx || i >= temp.width + sx)
        {
            //pimpl->dev.padded_templates[index].x = 0.0f;
            //pimpl->dev.padded_templates[index].y = 0.0f;
        }
        else if (j < sy || j >= temp.height + sy)
        {
            //pimpl->dev.padded_templates[index].x = 0.0f;
            //pimpl->dev.padded_templates[index].y = 0.0f;
        }
        else
        {
            float cur = temp.data[n * size + j * temp.width + i]; //像素点的值
            pimpl->dev.padded_templates[index].x = cur;
            //pimpl->dev.padded_templates[index].y = 0.0f;
        }
    }
    
    
    
        
    //std::printf("Passed assignment\n");
    cudaMemPrefetchAsync(pimpl->dev.padded_templates.get(), N_pixel * sizeof(cufftComplex), pimpl->dev.id, pimpl->dev.stream);
    //cudaMemcpyAsync(pimpl->dev.padded_templates.get(), padded_templates.data(), sizeof(cufftComplex) * N_pixel, cudaMemcpyHostToDevice, pimpl->dev.stream);

    cudaStreamSynchronize(pimpl->dev.stream);

    //auto padded_templates = std::make_unique<cufftComplex[]>(N_pixel);
    */
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
    float std = sqrt(static_cast<float>(sum_s2)  / image_size - avg * avg); // 标准差
    float up_bound = avg + 6 * std; // 以6倍标准差为上界
    float low_bound = avg - 6 * std; // 以6倍标准差为下界
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

void SearchNorm::PreprocessTemplate() 
{
    // **************************************************************
    // apply whitening filter and do ift
    // input: Padded IMAGE (Real SPACE)
    // output: IMAGE_whiten (Fourier SPACE in RI)
    // **************************************************************
    cudaMemsetAsync(pimpl->dev.ra.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);
    cudaMemsetAsync(pimpl->dev.rb.get(), 0, batch_size * RA_SIZE * sizeof(float), pimpl->dev.stream);
    
    //cudaEventQuery(start);

    // Inplace FFT 傅里叶变换
    cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(), pimpl->dev.padded_templates.get(), CUFFT_FORWARD);

    // CUFFT will enlarge VALUE to N times. Restore it
    //scale<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), N_pixel, padded_template_size);

    // 将直角坐标转换为极坐标
    SQRSum_by_circle<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), padding_size, padding_size, N_pixel);

    /*
    auto h_ra = std::make_unique<float[]>(batch_size * RA_SIZE);
    auto h_rb = std::make_unique<float[]>(batch_size * RA_SIZE);

    cudaMemcpyAsync(h_ra.get(), pimpl->dev.ra.get(), sizeof(float) * batch_size*RA_SIZE, cudaMemcpyDeviceToHost, pimpl->dev.stream);
    cudaMemcpyAsync(h_rb.get(), pimpl->dev.rb.get(), sizeof(float) * batch_size*RA_SIZE, cudaMemcpyDeviceToHost, pimpl->dev.stream);

    cudaStreamSynchronize(pimpl->dev.stream);
    std::cout << h_ra[3] << " " << h_rb[3] << std::endl;
    saveDataToFile(h_ra.get(), batch_size*RA_SIZE, "./f_square.txt");
    saveDataToFile(h_rb.get(), batch_size*RA_SIZE, "./f_count.txt");
    */

    // Whiten at fourier space
    // 将极坐标转换为直角坐标
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
    
    ap2ri<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), N_pixel);
    cudaMemsetAsync(pimpl->dev.means.get(), 0, batch_size * sizeof(float), pimpl->dev.stream);
    
    // 傅里叶逆变换
    cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.padded_templates.get(), pimpl->dev.padded_templates.get(), CUFFT_INVERSE);

    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::WriteTemplates(const std::string & filename, int nx, int ny, int nz, float pixel_size)
{
    auto processed_templates = std::make_unique<cufftComplex[]>(N_pixel);
    cudaMemcpyAsync(processed_templates.get(), pimpl->dev.padded_templates.get(), sizeof(cufftComplex) * N_pixel, cudaMemcpyDeviceToHost, pimpl->dev.stream);
    cudaStreamSynchronize(pimpl->dev.stream);

    auto output_templates = std::make_unique<float[]>(N_pixel);
    for (int i = 0; i < N_pixel; i++)
    {
        output_templates[i] = (float) processed_templates[i].x;
    }

    write_mrcs(filename, output_templates.get(), nx, ny, nz, pixel_size);
}

void SearchNorm::PreprocessImage(const Image & img) 
{
    int nblocks = (image_size - 1) / BLOCK_SIZE + 1;

    cudaMemsetAsync(pimpl->dev.padded_image.get(), 0, N_pixel1 * sizeof(cufftComplex), pimpl->dev.stream);

    float2Complex<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.image.get(), image_size);
    // fft inplace
    cufftExecC2C(pimpl->dev.fft.raw_image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(), CUFFT_FORWARD);
    //scale<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), image_size, image_size);

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

    if (pimpl->dev.k)
    {
        if (pimpl->dev.k1)
        {
            whiten_filter_weight_Img<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, para, pimpl->dev.k.get(), pimpl->dev.fsc.get(), fsc_size, pimpl->dev.k1.get(), pimpl->dev.n_k.get(), nk_size);
        }
        else
        {
            int mode = 1; // FSC only
            whiten_filter_weight_Img<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, para, pimpl->dev.k.get(), pimpl->dev.fsc.get(), fsc_size, mode);
        }
    }
    else
    {
        if (pimpl->dev.k1)
        {
            int mode = 2; // n(k) only
            whiten_filter_weight_Img<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, para, pimpl->dev.k1.get(), pimpl->dev.n_k.get(), nk_size, mode);
        }
        else
        {
            whiten_filter_weight_Img<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), pimpl->dev.ra.get(), pimpl->dev.rb.get(), nx, ny, para);
        }
    }

    // 0Hz -> 0
    //set_0Hz_to_0_at_RI<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get());

    // ifft inplace
    cufftExecC2C(pimpl->dev.fft.raw_image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(), CUFFT_INVERSE);
    Complex2float<<<nblocks, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.image.get(), pimpl->dev.padded_image.get(), image_size);

    cudaStreamSynchronize(pimpl->dev.stream);
    //pimpl->dev.ra = nullptr;
    //pimpl->dev.rb = nullptr;
}

void SearchNorm::SplitImage() 
{
    // split Image into blocks with overlap
    split_IMG<<<grid_size1, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.image.get(), pimpl->dev.padded_image.get(), pimpl->dev.offset_x.get(), pimpl->dev.offset_y.get(), nx, ny, padding_size, block_x, N_pixel1);

    // do normalize to all subIMGs
    // Inplace FFT
    cufftExecC2C(pimpl->dev.fft.image, pimpl->dev.padded_image.get(), pimpl->dev.padded_image.get(), CUFFT_FORWARD);
    // Scale IMG to normal size
    //scale<<<grid_size1, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_image.get(), N_pixel1, padded_template_size);
    /*
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
    */
    if (pimpl->dev.split_image)
    {
        fft_shift_pad<<<grid_size3, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.split_image.get(), pimpl->dev.padded_image.get(), ft_padding, padding_size, N_pixel3);    
    }

    cudaMemsetAsync(pimpl->dev.CCG_sum.get(), 0, sizeof(cufftComplex) * N_pixel1, pimpl->dev.stream);
    cudaStreamSynchronize(pimpl->dev.stream);
    //pimpl->dev.image = nullptr;
}

void SearchNorm::RotateTemplate(float euler3) 
{
    if (pimpl->dev.rotated_templates)
    {// 旋转所有模板
        rotate_subIMG<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.rotated_templates.get(), euler3, padding_size, N_pixel);
        // 施加mask，将边缘点设为0，soft edge mask
        apply_mask<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.rotated_templates.get(), para.d_m, para.edge_width, padding_size, N_pixel);
        // 求和，求平方和
        compute_sum_sqr<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.rotated_templates.get(), pimpl->dev.reduction_buf.get(), N_pixel);
    }
    else
    {
        rotate_subIMG<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.padded_templates.get(), pimpl->dev.CCG_buf.get(), euler3, padding_size, N_pixel);
        // 施加mask，将边缘点设为0，soft edge mask
        apply_mask<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), para.d_m, para.edge_width, padding_size, N_pixel);
        // 求和，求平方和
        compute_sum_sqr<<<grid_size, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), pimpl->dev.reduction_buf.get(), N_pixel);

    }
    
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

    if (pimpl->dev.rotated_templates)
    {
        substract_by_mean<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.rotated_templates.get(), padded_template_size, pimpl->dev.means.get(), N_pixel);
        cudaMemcpyAsync(pimpl->dev.means.get(), infile_sqr, sizeof(float) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
        divided_by_var<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.rotated_templates.get(), padded_template_size, pimpl->dev.means.get(), N_pixel);
        
        cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.rotated_templates.get(), pimpl->dev.rotated_templates.get(), CUFFT_FORWARD);
        //scale<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.rotated_templates.get(), N_pixel, padded_template_size);
        fft_shift_pad<<<grid_size2, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), pimpl->dev.rotated_templates.get(), ft_padding, padding_size, N_pixel2);    
    }
    else
    {
        substract_by_mean<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), padded_template_size, pimpl->dev.means.get(), N_pixel);
        cudaMemcpyAsync(pimpl->dev.means.get(), infile_sqr, sizeof(float) * batch_size, cudaMemcpyHostToDevice, pimpl->dev.stream);
        divided_by_var<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), padded_template_size, pimpl->dev.means.get(), N_pixel);

        cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG_buf.get(), pimpl->dev.CCG_buf.get(), CUFFT_FORWARD);

        //scale<<<grid_size, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_buf.get(), N_pixel, padded_template_size);
    }
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::ComputeCCGSum() 
{
    // compute score for each block
    for (int j = 0; j < block_y; ++j) 
    {
        for (int i = 0; i < block_x; ++i) 
        {
            if (pimpl->dev.split_image)
            {
                // compute CCG
                compute_corner_CCG<<<grid_size2, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.CCG_buf.get(), pimpl->dev.split_image.get(), ft_padding, i + j * block_x, N_pixel2);
                // Inplace IFT
                cufftExecC2C(pimpl->dev.fft.templates1, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(), CUFFT_INVERSE);
            }
            else
            {
                /*
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, pimpl->dev.stream);
                */
                compute_corner_CCG<<<grid_size2, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.CCG_buf.get(), pimpl->dev.padded_image.get(), ft_padding, i + j * block_x, N_pixel2);
                /*
                cudaEventRecord(stop, pimpl->dev.stream);
                cudaEventSynchronize(stop);
                float elapsed_time;
                cudaEventElapsedTime(&elapsed_time, start, stop);
                printf("Time = %g ms.\n", elapsed_time);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                */

                cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(), CUFFT_INVERSE);
            }
            // compute avg/variance
            add_CCG_to_sum<<<(ft_padded_size - 1)/ BLOCK_SIZE + 1, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_sum.get(), pimpl->dev.CCG.get(), ft_padded_size, batch_size, i + j * block_x);   
        }
    }
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::ComputeCCGMean() 
{
    set_CCG_mean<<<grid_size3, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_sum.get(), N_pixel3, batch_size * (360 / phi_step));
    cudaStreamSynchronize(pimpl->dev.stream);
}

void SearchNorm::PickParticles(std::vector<float>& scores) 
{
    scores.clear();

    // compute score for each block
    for (int j = 0; j < block_y; j++) 
    {
        for (int i = 0; i < block_x; i++) 
        {
            if (pimpl->dev.split_image)
            {
                // compute CCG
                compute_corner_CCG<<<grid_size2, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.CCG_buf.get(), pimpl->dev.split_image.get(), ft_padding, i + j * block_x, N_pixel2);
                // Inplace IFT
                cufftExecC2C(pimpl->dev.fft.templates1, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(), CUFFT_INVERSE);
                // update CCG with avg/var
                update_CCG<<<grid_size2, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_sum.get(), pimpl->dev.CCG.get(), ft_padded_size, i + j * block_x, N_pixel2);

                // find peak in each block
                get_peak_pos<<<grid_size2, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.reduction_buf2.get(), ft_padded_size, N_pixel2);

                cudaMemcpyAsync(pimpl->host.reduction_buf2.get(), pimpl->dev.reduction_buf2.get(), 2 * sizeof(float) * grid_size2, cudaMemcpyDeviceToHost, pimpl->dev.stream);
                cudaStreamSynchronize(pimpl->dev.stream);

                // After Reduction -> compute mean for each image
                for (int k = 0; k < grid_size2; k++) 
                {
                    if (pimpl->host.reduction_buf2[2 * k] >= para.thresh) 
                    {
                        float score = pimpl->host.reduction_buf2[2 * k];
                        float centerx, centery;
                        centerx = block_offsets_x[i] + int(std::round(((int)pimpl->host.reduction_buf2[2 * k + 1] % ft_padding) * ((padding_size-1) / static_cast<float>(ft_padding-1))));
                        centery = block_offsets_y[j] + int(std::round(((int)pimpl->host.reduction_buf2[2 * k + 1] / ft_padding) * ((padding_size-1) / static_cast<float>(ft_padding-1))));
                        if (centerx >= para.d_m / 2 && centerx < nx - para.d_m / 2 && centery >= para.d_m / 2 && centery < ny - para.d_m / 2) 
                        {
                            scores.emplace_back(score);
                            scores.emplace_back(centerx);
                            scores.emplace_back(centery);
                            scores.emplace_back(k * BLOCK_SIZE / ft_padded_size);
                        }
                    }
                }
            }
            else
            {
                compute_corner_CCG<<<grid_size2, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.CCG_buf.get(), pimpl->dev.padded_image.get(), ft_padding, i + j * block_x, N_pixel2);
                // Inplace IFT
                cufftExecC2C(pimpl->dev.fft.templates, pimpl->dev.CCG.get(), pimpl->dev.CCG.get(), CUFFT_INVERSE);
                update_CCG<<<grid_size2, BLOCK_SIZE, 0, pimpl->dev.stream>>>(pimpl->dev.CCG_sum.get(), pimpl->dev.CCG.get(), ft_padded_size, i + j * block_x, N_pixel2);

                // find peak in each block
                get_peak_pos<<<grid_size2, BLOCK_SIZE, 2 * BLOCK_SIZE * sizeof(float), pimpl->dev.stream>>>(pimpl->dev.CCG.get(), pimpl->dev.reduction_buf.get(), ft_padded_size, N_pixel2);

                cudaMemcpyAsync(pimpl->host.reduction_buf.get(), pimpl->dev.reduction_buf.get(), 2 * sizeof(float) * grid_size2, cudaMemcpyDeviceToHost, pimpl->dev.stream);
                cudaStreamSynchronize(pimpl->dev.stream);

                // After Reduction -> compute mean for each image
                for (int k = 0; k < grid_size2; k++) 
                {
                    if (pimpl->host.reduction_buf[2 * k] >= para.thresh) 
                    {
                        float score = pimpl->host.reduction_buf[2 * k];
                        float centerx, centery;
                        centerx = block_offsets_x[i] + (int)pimpl->host.reduction_buf[2 * k + 1] % padding_size;
                        centery = block_offsets_y[j] + (int)pimpl->host.reduction_buf[2 * k + 1] / padding_size;
                        if (centerx >= para.d_m / 2 && centerx < nx - para.d_m / 2 && centery >= para.d_m / 2 && centery < ny - para.d_m / 2) 
                        {
                            scores.emplace_back(score);
                            scores.emplace_back(centerx);
                            scores.emplace_back(centery);
                            scores.emplace_back(k * BLOCK_SIZE / ft_padded_size);
                        }
                    }
                }
            }
        }
    }
}

void SearchNorm::OutputScore(std::string & output, std::vector<float> & scores, const Image & input) 
{
    int line_count = 0;
    char buf[1024];
    std::fstream out(output, std::ios::out | std::ios::app);
    std::filesystem::path filePath = output;
    if (filePath.extension() == ".star")
    {
        float df1 = -para.dfu * 10000;
        float df2 = -para.dfv * 10000;
        for (int i = 0; i < scores.size(); i += 4) 
        {
            float score = scores[i];
            float centerx = scores[i + 1];
            float centery = scores[i + 2];
            size_t j = scores[i + 3];
            std::snprintf(buf, 1024, "%s %f %f %f %f %f %f %f %f %d #%f\n", input.rpath.c_str(), centerx, centery, df1, df2, para.dfang, euler.euler2[j], euler.euler1[j], euler.euler3[j], 1, score);
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
            std::snprintf(buf, 1024, "%d\t%s\tdefocus=%f\tdfdiff=%f\tdfang=%f\teuler=%f,%f,%f\tcenter=%f,%f\tscore=%f\n", input.unused, input.rpath.c_str(), para.defocus, para.dfdiff, para.dfang+90, euler.euler1[j], euler.euler2[j], euler.euler3[j], centerx, centery, score); 
            out << buf; 
            //std::cout << "Writing to the output file.\n";
            ++line_count; 
        }
    }
}

void SearchNorm::OutputScore(std::string & output, std::vector<float> & scores, float euler3, const Image & input, int & line_count) 
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

void SearchNorm::work_verbose(const Image & image, std::string & output) 
{
    using namespace std;
    int lc = 0;
    int & line_count = lc;
    vector<float> scores;
    auto params = image.p;

    SetParams(params);
    printf("Device %d: Image: %s, (%zu, %zu)\n", pimpl->dev.id, image.rpath.c_str(), params.width, params.height);
    printf("Device %d: Loading image\n", pimpl->dev.id);
    LoadImage(image);

    printf("Device %d: Preprocessing image\n", pimpl->dev.id);
    PreprocessImage(image);
    printf("Device %d: Splitting image\n", pimpl->dev.id);
    SplitImage();
    printf("Device %d: Computing the avgs and vars of all CCGs, euler3 in [0, 360), step = %.3f\n", pimpl->dev.id, phi_step);
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
    printf("Device %d: Updating CCGs and computing scores, euler3 in [0, 360), step = %.3f\n", pimpl->dev.id, phi_step);
    for (float euler3 = 0.0f; euler3 < 360.0f; euler3 += phi_step) 
    {
        RotateTemplate(euler3);
        PickParticles(scores);
        OutputScore(output, scores, euler3, image, line_count);
        if (static_cast<int>(euler3) % 72 == 0) 
        {
            printf(":) ");
            fflush(stdout);
        }
    }
    
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