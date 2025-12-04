#include "constants.h"
#include "kernels.cuh"

__global__ void UpdateSigma(cufftComplex * d_templates, float * d_buf, const int N) 
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;

    sdata[tid] = d_templates[i].x;
    sdata[tid + blockDim.x] = d_templates[i].x * d_templates[i].x;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            // sum of data[i] & data[i]^2
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }
    if (tid == 0) 
    {
        d_buf[blockIdx.x * 2] = sdata[0];
        d_buf[blockIdx.x * 2 + 1] = sdata[blockDim.x];
    }
}

__global__ void generate_mask(int l, cufftComplex * mask, float r, float * res, float up, float low) 
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    int image_size = l * l;
    int local_id = i % image_size;
    int x = local_id % l;
    int y = local_id / l;

    // Dis^2 between (x,y) and center (l/2,l/2)
    float rr = (x - l / 2) * (x - l / 2) + (y - l / 2) * (y - l / 2);
    // 将半径位于上下限之间的点的mask记为1
    if (rr >= low && rr <= up) 
    {
        mask[i].x = 1;
        mask[i].y = 0;
    }

    // reduction for the number of non-zero digits
    sdata[tid] = mask[i].x;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) 
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // write result for this block to global mem 每个block中的非零元素数量
    if (tid == 0) res[blockIdx.x] = sdata[0];
}

__global__ void multiCount_dot(int l, cufftComplex * mask, cufftComplex * d_templates, float * constants, float * res) 
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    int image_size = l * l;
    int image_id = i / image_size;

    // Multi constant 1/non-zeros
    if (constants[image_id] != 0) mask[i].x *= 1.0 / constants[image_id];

    // reduction for dot
    sdata[tid] = mask[i].x * d_templates[i].x;
    __syncthreads();

    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) 
    {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // use res to store dot results
    if (tid == 0) res[blockIdx.x] = sdata[0];
}

__global__ void scale_each(int l, cufftComplex * d_templates, float * d_sigmas, const int N) 
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // int image_size = l * l;
    int image_id = i / (l * l);

    if (d_sigmas[image_id] - 0 < EPS && d_sigmas[image_id] - 0 > -EPS) return;
    // 模板减去噪声，然后除以标准差
    d_templates[i].x = d_templates[i].x / d_sigmas[image_id];
}

__global__ void SQRSum_by_circle(cufftComplex * data, float * ra, float * rb, int nx, int ny, const unsigned int N) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int pixel_id = i % (nx * ny);
    int x = pixel_id % nx;
    int y = pixel_id / nx;

    float tmp;
    // 将直角坐标转换为极坐标
    tmp = hypotf(data[i].x, data[i].y);
    if (data[i].x == 0) data[i].y = 0;
    else data[i].y = atan2(data[i].y, data[i].x);
    data[i].x = tmp;

    //if (x > nx / 2) return; 

    // calculate the number of point with fixed distance ('r') from center
    // -1是去掉中心点
    int r = floor(hypotf(min(y, ny - y), min(x, nx - x)) + 0.5);

    if (r < max(nx, ny) / 2 && r >= 0) 
    {
        // Add offset
        r += RA_SIZE * (i / (nx * ny)); // 每张照片5000个点
        atomicAdd(&ra[r], data[i].x * data[i].x); // 对半径为r的圆上每个点的模平方求和
        atomicAdd(&rb[r], 1.0);
    }
}

__global__ void whiten_Tmp(cufftComplex * data, float * ra, float * rb, int l, const unsigned int N) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int pixel_id = i % (l * l);
    int x = pixel_id % l;
    int y = pixel_id / l;
    int r = floor(hypotf(min(y, l - y), min(x, l - x)) + 0.5); // 计算每个点离四个角的距离

    if (r < l / 2 && r >= 0) 
    {
        // Add offset
        r += RA_SIZE * (i / (l * l));
        //float fb_infile = ra[r] / rb[r];
        data[i].x = data[i].x / sqrtf(ra[r] / rb[r]); // 每个点的模除以模平方圆平均后开根号的值
    }
}

__global__ void whiten_filter_weight_Img(cufftComplex * data, float * ra, float * rb, int nx, int ny, Parameters para, float * k, float * fsc, int size, float * k1, float * nk, int size1) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (nx * ny)) return;

    int pixel_id = i % (nx * ny);
    int x = pixel_id % nx;
    int y = pixel_id / nx;

    float dx = min(x, nx - x);
    float dy = min(y, ny - y);
    int r_round = floor(hypotf(dx, dy) + 0.5);
    float r = hypotf(dx, dy);
    float ss = sqrtf((dx * dx / (float)(nx * nx) + dy * dy / (float)(ny * ny)) / (para.apix * para.apix));
    int l = max(nx, ny);
    float v, signal, Ncurve;

    // whiten
    if (r_round < l / 2 && r_round >= 0) 
    {
        data[i].x = data[i].x / sqrt(ra[r_round] / rb[r_round]);
        if (r > (l * para.apix / 6)) data[i].x = data[i].x * exp(-100 * ss * ss);
    }

    // low pass
    if (r <= l * para.apix / para.highres && r >= l * para.apix / para.lowres) {}
    else if (r > l * para.apix / para.highres && r < l * para.apix / para.highres + para.edge_width) 
    {
        data[i].x = data[i].x * (0.5 * cosf(PI * (r - l * para.apix / para.highres) / para.edge_width) + 0.5);
    } 
    else if (r > (l * para.apix / para.lowres - para.edge_width) && r < l * para.apix / para.lowres) 
    {
        data[i].x = data[i].x * (0.5 * cosf(PI * (l * para.apix / para.lowres - r) / para.edge_width) + 0.5);
    } 
    else 
    {
        data[i].x = 0;
    }

    // apply weighting function
    if (r <= l / 2 && r > 0) 
    {
        v = CTF(x, y, nx, ny, para.apix, para.dfu, para.dfv, para.dfdiff, para.dfang, para.lambda, para.cs, para.ampconst, 1);
        signal = exp(para.bfactor * ss * ss + para.bfactor2 * ss + para.bfactor3);
        Ncurve = exp(para.a * ss * ss + para.b * ss + para.b2) / signal;
        float fsc_interp = interp_fsc(k, fsc, size, ss);
        float nk_interp = interp_nk(k1, nk, size1, ss);
        float w_fsc = (2 * fsc_interp / (1 + fsc_interp)) * (2 * fsc_interp / (1 + fsc_interp));
        data[i].x = data[i].x * w_fsc * v / (Ncurve + nk_interp * para.kk * v * v);
    }

    // ap2ri
    float tmp = data[i].x * sinf(data[i].y);
    data[i].x = data[i].x * cosf(data[i].y);
    data[i].y = tmp;
}

__global__ void whiten_filter_weight_Img(cufftComplex * data, float * ra, float * rb, int nx, int ny, Parameters para, float * k, float * fsc, int size, int mode) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (nx * ny)) return;

    int pixel_id = i % (nx * ny);
    int x = pixel_id % nx;
    int y = pixel_id / nx;

    float dx = min(x, nx - x);
    float dy = min(y, ny - y);
    int r_round = floor(hypotf(dx, dy) + 0.5);
    float r = hypotf(dx, dy);
    float ss = sqrtf((dx * dx / (float)(nx * nx) + dy * dy / (float)(ny * ny)) / (para.apix * para.apix));
    int l = max(nx, ny);
    float v, signal, Ncurve;

    // whiten
    if (r_round < l / 2 && r_round >= 0) 
    {
        data[i].x = data[i].x / sqrt(ra[r_round] / rb[r_round]);
        //if (r > (l * para.apix / 6)) data[i].x = data[i].x * exp(-100 * ss * ss);
    }

    // low pass
    if (r <= l * para.apix / para.highres && r >= l * para.apix / para.lowres) {}
    else if (r > l * para.apix / para.highres && r < l * para.apix / para.highres + para.edge_width) 
    {
        data[i].x = data[i].x * (0.5 * cosf(PI * (r - l * para.apix / para.highres) / para.edge_width) + 0.5);
    } 
    else if (r > (l * para.apix / para.lowres - para.edge_width) && r < l * para.apix / para.lowres) 
    {
        data[i].x = data[i].x * (0.5 * cosf(PI * (l * para.apix / para.lowres - r) / para.edge_width) + 0.5);
    } 
    else 
    {
        data[i].x = 0;
    }

    // apply weighting function
    if (r <= l / 2 && r > 0) 
    {
        v = CTF(x, y, nx, ny, para.apix, para.dfu, para.dfv, para.dfdiff, para.dfang, para.lambda, para.cs, para.ampconst, 1);
        signal = exp(para.bfactor * ss * ss + para.bfactor2 * ss + para.bfactor3);
        Ncurve = exp(para.a * ss * ss + para.b * ss + para.b2) / signal;
        if (mode == 1)
        {
            float fsc_interp = interp_fsc(k, fsc, size, ss);
            float w_fsc = (2 * fsc_interp / (1 + fsc_interp)) * (2 * fsc_interp / (1 + fsc_interp));    
            data[i].x = data[i].x * w_fsc * v / (Ncurve + para.kk * v * v);
        }
        else
        {
            float nk_interp = interp_nk(k, fsc, size, ss);
            data[i].x = data[i].x * v / (Ncurve + nk_interp * para.kk * v * v);
        }
    }

    // ap2ri
    float tmp = data[i].x * sinf(data[i].y);
    data[i].x = data[i].x * cosf(data[i].y);
    data[i].y = tmp;
}

__global__ void whiten_filter_weight_Img(cufftComplex * data, float * ra, float * rb, int nx, int ny, Parameters para) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (nx * ny)) return;

    int pixel_id = i % (nx * ny);
    int x = pixel_id % nx;
    int y = pixel_id / nx;

    float dx = min(x, nx - x);
    float dy = min(y, ny - y);
    float r = hypotf(dx, dy);
    int r_round = floor(r + 0.5);
    float ss = sqrtf((dx * dx / (float)(nx * nx) + dy * dy / (float)(ny * ny)) / (para.apix * para.apix));
    int l = max(nx, ny);
    float v, signal, Ncurve;

    // whiten
    if (r_round < l / 2 && r_round >= 0) 
    {
        data[i].x = data[i].x / sqrt(ra[r_round] / rb[r_round]);
        //if (r > (l * para.apix / 6)) data[i].x = data[i].x * exp(-100 * ss * ss);
    }

    // low pass
    if (r <= l * para.apix / para.highres && r >= l * para.apix / para.lowres) {}
    else if (r > l * para.apix / para.highres && r < l * para.apix / para.highres + para.edge_width) 
    {
        data[i].x = data[i].x * (0.5 * cosf(PI * (r - l * para.apix / para.highres) / para.edge_width) + 0.5);
    } 
    else if (r > (l * para.apix / para.lowres - para.edge_width) && r < l * para.apix / para.lowres) 
    {
        data[i].x = data[i].x * (0.5 * cosf(PI * (l * para.apix / para.lowres - r) / para.edge_width) + 0.5);
    } 
    else 
    {
        data[i].x = 0;
    }

    // apply weighting function
    if (r <= l / 2 && r > 0) 
    {
        v = CTF(x, y, nx, ny, para.apix, para.dfu, para.dfv, para.dfdiff, para.dfang, para.lambda, para.cs, para.ampconst, 1);
        signal = exp(para.bfactor * ss * ss + para.bfactor2 * ss + para.bfactor3);
        Ncurve = exp(para.a * ss * ss + para.b * ss + para.b2) / signal;
        data[i].x = data[i].x * v / (Ncurve + para.kk * v * v);
    }

    // ap2ri
    float tmp = data[i].x * sinf(data[i].y);
    data[i].x = data[i].x * cosf(data[i].y);
    data[i].y = tmp;
}

__global__ void set_0Hz_to_0_at_RI(cufftComplex* data) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1) return;

    data[i].x = 0; // 将0频处的信号取0
    data[i].y = 0;
}

__global__ void apply_mask(cufftComplex * data, float d_m, float edge_width, int l, const unsigned int N) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int pixel_id = i % (l*l);
    int x = pixel_id % l;
    int y = pixel_id / l;
    d_m = 1.5 * d_m; // 蛋白质直径，单位为像素
    float r = hypotf(x - l / 2, y - l / 2);
    if (r > (d_m / 2 + edge_width)) 
    {
        data[i].x = 0;
    } 
    else if (r > d_m / 2 && r < (d_m / 2 + edge_width)) 
    {
        float d = 0.5 * cosf(PI * (r - d_m / 2) / edge_width) + 0.5;
        data[i].x *= d;
    }
}

__global__ void apply_weighting_function(cufftComplex * data, int l, Parameters para, const unsigned int N) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    //int l = padding_size;
    //int image_size = l * l;
    int pixel_id = i % (l*l);
    int x = pixel_id % l;
    int y = pixel_id / l;

    // low pass
    float r = hypotf(min(y, l - y), min(x, l - x));
    //int r_round = floor(r + 0.5) - 1;
    // 在最大/最小空间频率处，乘上软边界（宽度为8像素）
    if (r <= l * para.apix / para.highres && r >= l * para.apix / para.lowres) {} 
    else if (r > l * para.apix / para.highres && r < l * para.apix / para.highres + para.edge_width) 
    {
        data[i].x = data[i].x * (0.5 * cosf(PI * (r - l * para.apix / para.highres) / para.edge_width) + 0.5);
    } 
    else if (r < l * para.apix / para.lowres && r > (l * para.apix / para.lowres - para.edge_width)) 
    {
        data[i].x = data[i].x * (0.5 * cosf(PI * (l * para.apix / para.lowres - r) / para.edge_width) + 0.5);
    } 
    else
    {
        data[i].x = 0;
    }
}

__device__ float interp_fsc(const float* keys, const float* values,
    int size, const float query) 
{   
    // 二分查找
    int left = 0;
    int right = size-1;
    while(left <= right) 
    {
        int mid = (left + right) / 2;
        if(keys[mid] < query) 
        {
            left = mid + 1;
        } 
        else 
        {
            right = mid - 1;
        }
    }

    const int idx_l = left - 1;
    const int idx_r = left;
    
    const float key_l = keys[idx_l];
    const float key_r = keys[idx_r];
    const float val_l = values[idx_l];
    const float val_r = values[idx_r];

    // 计算插值比例
    const float delta = key_r - key_l;
    const float t = (delta != 0.0f) ? ((query - key_l) / delta) : 0.5f;

    // 线性插值
    float result = val_l + t * (val_r - val_l);
    return result;
}

__device__ float interp_nk(const float* keys, const float* values,
    int size, const float query) 
{   
    // 二分查找
    int left = 0;
    int right = size-1;
    if (query < keys[left]) {return 1;}
    if (query > keys[right]) {return 1;}
    else
    {
        while(left <= right) 
        {
            int mid = (left + right) / 2;
            if(keys[mid] < query) 
            {
                left = mid + 1;
            } 
            else 
            {
                right = mid - 1;
            }
        }

        const int idx_l = left - 1;
        const int idx_r = left;
        
        const float key_l = keys[idx_l];
        const float key_r = keys[idx_r];
        const float val_l = values[idx_l];
        const float val_r = values[idx_r];

        // 计算插值比例
        const float delta = key_r - key_l;
        const float t = (delta != 0.0f) ? ((query - key_l) / delta) : 0.5f;

        // 线性插值
        float result = val_l + t * (val_r - val_l);
        return result;
    }
}

__device__ float CTF(int x1, int y1, int nx, int ny, float apix, float dfu, float dfv, float dfdiff, float dfang, float lambda, float cs, float ampconst, int mode) 
{
    float v, ss, ag, gamma, df_ast;
    int x, y;
    x = x1 > nx / 2 ?  x1 - nx : x1;
    y = y1 > ny / 2 ?  ny - y1 : -y1;
    ss = (x * x / (float)(nx * nx) + y * y / (float)(ny * ny)) / (apix * apix); // g
    ag = atan2(float(y), float(x)); // alpha_g

    df_ast = 0.5 * (dfu + dfv + 2 * dfdiff * cosf(2 * (dfang * PI / 180 - ag)));
    gamma = PI * lambda * ss * (-cs * 5e6 * lambda * lambda * ss + df_ast * 1e4);
    if (mode == 0) 
    {
        v = (sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) + ampconst * cosf(gamma)) > 0 ? 1.0 : -1.0;  // do phase flipping
    } 
    else if (mode == 2) 
    {
        v = fabs(sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) + ampconst * cosf(gamma));  // return abs ctf value
    } 
    else 
    {
        v = sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) + ampconst * cosf(gamma);  // return ctf value
    }
    return v;
}

__device__ float CTF_AST(int x1, int y1, int nx, int ny, float apix, float dfu, float dfv, float dfdiff, float dfang, float lambda, float cs, float ampconst, int mode) 
{
    float v, ss, ag, gamma, df_ast;
    y1 = x1 > nx / 2 ? ny - y1 : y1;
    x1 = min(x1, nx - x1);
    float dx = min(x1, nx - x1);
    float dy = y1 - ny / 2;
    ss = (dx * dx / (float)(nx * nx) + dy * dy / (float)(ny * ny)) / (apix * apix); // g
    ag = atan2(float(y1 - ny / 2), float(x1)); // alpha_g

    df_ast = 0.5 * (dfu + dfv + 2 * dfdiff * cosf(2 * (dfang * PI / 180 - ag)));
    gamma = -2 * PI * (cs * 2.5e6 * lambda * lambda * lambda * ss * ss + df_ast * 5000.0 * lambda * ss);
    if (mode == 0) 
    {
        v = (sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) + ampconst * cosf(gamma)) > 0 ? 1.0 : -1.0;  // do phase flipping
    } 
    else if (mode == 2) 
    {
        v = fabs(sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) + ampconst * cosf(gamma));  // return abs ctf value
    } 
    else 
    {
        v = (sqrtf(1.0 - ampconst * ampconst) * sinf(gamma) + ampconst * cosf(gamma));  // return ctf value
    }
    return v;
}

__global__ void compute_area_sum_ofSQR(cufftComplex * data, float * res, int l, const int N) 
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;

    int pixel_id = i % (l * l);
    int x = pixel_id % l;
    int y = pixel_id / l;
    int r = floor(hypotf(min(y, l - y), min(x, l - x)) + 0.5);

    if (r < l / 2 && r >= 0 && x <= l / 2) 
    {
        sdata[tid] = data[i].x * data[i].x; // 将圆内点的模平方求和
        sdata[tid + blockDim.x] = 1;
    } 
    else 
    {
        sdata[tid] = 0;
        sdata[tid + blockDim.x] = 0;
    }
    __syncthreads();

    // if (tid < 512) {
    //   sdata[tid] += sdata[tid + 512];
    //   sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 512];
    // }
    // __syncthreads();
    // if (tid < 256) {
    //   sdata[tid] += sdata[tid + 256];
    //   sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 256];
    // }
    // __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            // sum of data[i] & data[i]^2
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }
    if (tid == 0) 
    {
        res[blockIdx.x * 2] = sdata[0];
        res[blockIdx.x * 2 + 1] = sdata[blockDim.x];
    }
}

__global__ void compute_sum_sqr(cufftComplex * data, float * res, const unsigned int N) 
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    auto tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;

    // int image_size = nx * ny;
    // int local_id = i % image_size;
    // int x = local_id % nx;
    // int y = local_id / nx;
    //int r = floor(hypotf(min(y, ny - y), min(x, nx - x)) + 0.5) - 1;

    sdata[tid] = data[i].x;
    sdata[tid + blockDim.x] = data[i].x * data[i].x;

    __syncthreads();

    // if (tid < 512) {
    //   sdata[tid] += sdata[tid + 512];
    //   sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 512];
    // }
    // __syncthreads();
    // if (tid < 256) {
    //   sdata[tid] += sdata[tid + 256];
    //   sdata[tid + blockDim.x] += sdata[tid + blockDim.x + 256];
    // }
    // __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            // sum of data[i] & data[i]^2
            sdata[tid] += sdata[tid + s];
            sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
        }
        __syncthreads();
    }

    // write result for this block
    if (tid == 0) 
    {
        res[2 * blockIdx.x] = sdata[0];
        res[2 * blockIdx.x + 1] = sdata[blockDim.x];
    }
}

__global__ void fft_shift_pad(cufftComplex * output, cufftComplex * input, int l, int l0, const int N, int width)
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (l != l0)
    {
        int pad_p = (l-l0)/2;
        int image_id = i / (l * l);
        int pixel_id = i % (l * l);
        int x = pixel_id % l;
        int y = pixel_id / l;

        int x_mid = (x + l/2) % l;
        int y_mid = (y + l/2) % l;

        bool in_valid_region = 
            (x_mid >= pad_p) && (x_mid < pad_p + l0) &&
            (y_mid >= pad_p)  && (y_mid < pad_p + l0);

        if (in_valid_region) 
        {
            // 计算对应的原数据坐标（在 FFTShift 后的布局中的位置）
            int x_shifted = x_mid - pad_p;  // 相对补零中心的坐标
            int y_shifted = y_mid - pad_p;

            // 应用 FFTShift 到原输入坐标（将中心坐标还原为原布局）
            int x_in = (x_shifted + l0/2) % l0;
            int y_in = (y_shifted + l0/2) % l0;

            float r = hypotf(min(x_in, l0-x_in), min(y_in, l0-y_in));
            if (r < l0/2 && r > l0/2 - width)
            {
                input[image_id * (l0*l0) + y_in * l0 + x_in].x *= (0.5 + 0.5 * cosf(PI*(r - (l0/2 - width))/width));
                input[image_id * (l0*l0) + y_in * l0 + x_in].y *= (0.5 + 0.5 * cosf(PI*(r - (l0/2 - width))/width));
            }
            if (r >= l0/2) 
            {
                input[image_id * (l0*l0) + y_in * l0 + x_in].x = 0;
                input[image_id * (l0*l0) + y_in * l0 + x_in].y = 0;
            }

            // 从输入读取数据
            output[i] = input[image_id * (l0*l0) + y_in * l0 + x_in];
        } else {
            // 补零区域直接写0
            output[i] = {0.0f, 0.0f};
        }
    }
    else
    {
        output[i] = input[i];
    }
}

__global__ void normalize(cufftComplex * data, int image_size, float * means, const int N) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int template_id = i / image_size;

    if (means[template_id] != 0) data[i].x = data[i].x / means[template_id];

    // ap2ri
    float tmp = data[i].x * sinf(data[i].y);
    data[i].x = data[i].x * cosf(data[i].y);
    data[i].y = tmp;
}

__global__ void divided_by_var(cufftComplex * data, int image_size, float * var, const unsigned int N) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    auto template_id = i / image_size;

    if (var[template_id] != 0) data[i].x = data[i].x / sqrtf(var[template_id]);
}

__global__ void substract_by_mean(cufftComplex * data, int image_size, float * means, const unsigned int N) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    auto template_id = i / image_size;

    if (means[template_id] != 0) data[i].x = data[i].x - means[template_id];
}

__global__ void rotate_IMG(float * d_image, float * d_rotated_image, float e, int nx, int ny) 
{
  float cose = cos(e * PI / 180);
  float sine = sin(e * PI / 180);
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  int j = id / nx, i = id % nx;
  float y = j - ny / 2, x = i - nx / 2;
  if (i >= nx || j >= ny) return;

  // Res of rotation from (x,y)
  float res = 0;

  //(x,y) rotate e with (nx/2,ny/2) (clockwise)
  float x2 = (cose * x + sine * y) + nx / 2;
  float y2 = (-sine * x + cose * y) + ny / 2;

  // Ouf of boundary after rotation
  if (x2 < 0 || x2 > nx - 1.0 || y2 < 0 || y2 > ny - 1.0)
    res = 0;
  else {
    int ii, jj;
    int k0, k1, k2, k3;
    float t, u, p0, p1, p2, p3;
    ii = floor(x2);
    jj = floor(y2);
    k0 = ii + jj * nx;
    k1 = k0 + 1;
    k2 = k0 + nx + 1;
    k3 = k0 + nx;

    // handle situation when ii,jj are out of boundary
    if (ii == nx - 1) {
      k1--;
      k2--;
    }
    if (jj == ny - 1) {
      k2 -= nx;
      k3 -= nx;
    }
    t = (x2 - (float)ii);
    u = (y2 - (float)jj);
    float tt = 1.0 - t;
    float uu = 1.0 - u;

    // bilinear interpolation of raw data (i,j)(i+1,j)(i,j+1)(i+1,j+1)
    p0 = d_image[k0] * tt * uu;
    p1 = d_image[k1] * t * uu;
    p3 = d_image[k3] * tt * u;
    p2 = d_image[k2] * t * u;
    res = p0 + p1 + p2 + p3;
  }

  // res <=> data[i+j*nx] after rotation
  d_rotated_image[id] = res;
}

__global__ void rotate_subIMG(cufftComplex * d_image, cufftComplex * d_rotated_image, float e, int l, const unsigned int N) 
{
    auto id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    float cose = cos(e * PI / 180);
    float sine = sin(e * PI / 180);

    int pixel_id = id % (l*l);
    int off = id - pixel_id; // 当前照片的（0，0）

    int i = pixel_id % l;
    int j = pixel_id / l;
    int nx = l, ny = l;
    float y = j - ny / 2, x = i - nx / 2;

    // Result of rotation from (x,y)
    float res_x = 0, res_y = 0;

    //(x,y) rotate e with (nx/2,ny/2) (counter-clock wise) 对应原函数的逆变换
    float x2 = (cose * x - sine * y) + nx / 2;
    float y2 = (sine * x + cose * y) + ny / 2;

    // Ouf of boundary after rotation
    if (x2 < 0 || x2 > nx - 1.0 || y2 < 0 || y2 > ny - 1.0) 
    {
        res_x = 0;
        res_y = 0;
    } 
    else 
    {
        int ii, jj;
        int k0, k1, k2, k3;
        float t, u, p0, p1, p2, p3;
        ii = floor(x2);
        jj = floor(y2);
        // 四个k点
        k0 = ii + jj * nx;
        k1 = k0 + 1;
        k2 = k0 + nx + 1;
        k3 = k0 + nx;

        // handle situation when ii,jj are out of boundary
        if (ii == nx - 1) 
        {
            k1 = k1 - 1;
            k2 = k2 - 1;
        }
        if (jj == ny - 1) 
        {
            k2 -= nx;
            k3 -= nx;
        }
        t = x2 - ii;
        u = y2 - jj;
        float tt = 1.0 - t;
        float uu = 1.0 - u;

        // bilinear interpolation of raw data (i,j)(i+1,j)(i,j+1)(i+1,j+1)
        p0 = d_image[off + k0].x * tt * uu;
        p1 = d_image[off + k1].x * t * uu;
        p3 = d_image[off + k3].x * tt * u;
        p2 = d_image[off + k2].x * t * u;
        res_x = p0 + p1 + p2 + p3;

        p0 = d_image[off + k0].y * tt * uu;
        p1 = d_image[off + k1].y * t * uu;
        p3 = d_image[off + k3].y * tt * u;
        p2 = d_image[off + k2].y * t * u;
        res_y = p0 + p1 + p2 + p3;
    }

    // res <=> data[i+j*nx] after rotation
    d_rotated_image[id].x = res_x;
    d_rotated_image[id].y = res_y;
}

__global__ void split_IMG(float * Ori, cufftComplex * IMG, int nx, int ny, int l, int bx, int overlap) 
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    int image_size = l * l;
    int image_id = i / image_size;
    int local_id = i % image_size;
    int x = local_id % l;
    int y = local_id / l;

    int tmp = l - overlap;

    int area_x_id = image_id % bx;
    int area_y_id = image_id / bx;
    int ori_x = area_x_id * tmp + x;
    int ori_y = area_y_id * tmp + y;

    if (ori_x >= nx || ori_y >= ny) return;
    IMG[i].x = Ori[ori_x + ori_y * nx];
}

__global__ void split_IMG(float * Ori, cufftComplex * IMG, int * block_off_x, int * block_off_y, int nx, int ny, int l, int bx, const int N) 
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int image_id = i / (l * l);
    int pixel_id = i % (l * l);
    int x = pixel_id % l;
    int y = pixel_id / l;

    int area_x_id = image_id % bx;
    int area_y_id = image_id / bx;
    int ori_x = block_off_x[area_x_id] + x;
    int ori_y = block_off_y[area_y_id] + y;

    if (ori_x >= nx || ori_y >= ny) return;
    IMG[i].x = Ori[ori_x + ori_y * nx]; // 将搜索区域从原照片中截出来
}

__global__ void compute_corner_CCG(cufftComplex * CCG, cufftComplex * Tl, cufftComplex * IMG, int l, int block_id, const unsigned int N) 
{
    // On this function, block means subimage splitted from IMG, not block ON GPU
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Local id corresponding to splitted IMG
    int pixel_id = i % (l*l);
    int local_x = pixel_id % l;
    int local_y = pixel_id / l;

    int off = block_id * (l*l);

    // Global ID in IMG
    int j = local_x + local_y * l + off;

    // CCG[i] = IMG[i]*template'[i]
    //  ' means conjugate
    CCG[i].x = (IMG[j].x * Tl[i].x + IMG[j].y * Tl[i].y);
    CCG[i].y = (IMG[j].y * Tl[i].x - IMG[j].x * Tl[i].y);

    // 施加相位偏移，使实空间中的峰值位于中心 Move center to around
    if ((local_x+local_y) % 2 == 1)
    {
        CCG[i].x *= -1;
        CCG[i].y *= -1;
    }
}

// compute the avg of CCG in all templates
__global__ void add_CCG_to_sum(cufftComplex * CCG_sum, cufftComplex * CCG, int image_size, int N, int block_id) 
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= image_size) return;

    // Area of rectangle, l^2
    //int interval = l * l;
    int off = block_id * image_size;

    // compute average & vairance
    for (int n = 0; n < N; n++) 
    {
        float cur = CCG[n * image_size + i].x; 
        //float cur = CCG[n * image_size + i].x; 
        CCG_sum[off + i].x += cur; // 在一个搜索框内，对所有模板的CC求和
        CCG_sum[off + i].y += (cur * cur);
    }
}

__global__ void set_CCG_mean(cufftComplex * CCG_sum, int N, int total_n) 
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    //float total_n = N_tmp * N_euler;

    float avg = CCG_sum[i].x / total_n;
    float std = sqrtf(CCG_sum[i].y / total_n - avg * avg);

    CCG_sum[i].x = avg;
    CCG_sum[i].y = std;
}

// update CCG val use avgeage & variance
__global__ void update_CCG(cufftComplex * CCG_sum, cufftComplex * CCG, int image_size, int block_id, unsigned int N) 
{
    // On this function,block means subimage splitted from IMG, not block ON GPU
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // Local id corresponding to splitted IMG
    int local_id = i % image_size;
    int off = block_id * image_size;

    float avg = CCG_sum[off + local_id].x;
    float std = CCG_sum[off + local_id].y;

    float cur = CCG[i].x;
    //float cur = CCG[i].x;
    CCG[i].x = std > 0 ? (cur - avg) / std : 0;
}

//"MAX" reduction for *odata : return max{odata[i]},i
//"SUM" reduction for *odata : return sum{odata[i]},sum{odata[i]^2}
__global__ void get_peak_and_SUM(cufftComplex* odata, float* res, int l, float d_m) {
  extern __shared__ float sdata[];
  // each thread loads one element from global to shared mem
  auto i = blockIdx.x * blockDim.x + threadIdx.x;

  int tid = threadIdx.x;
  int image_size = l * l;
  int local_id = i % image_size;
  int x = local_id % l;
  int y = local_id / l;

  sdata[tid] = odata[i].x;

  if (x < d_m / 4 || x > l - d_m / 4 || y < d_m / 4 || y > l - d_m / 4) sdata[tid] = 0;
  sdata[tid + blockDim.x] = local_id;
  sdata[tid + 2 * blockDim.x] = odata[i].x;
  sdata[tid + 3 * blockDim.x] = odata[i].x * odata[i].x;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      // find max
      if (sdata[tid + s] > sdata[tid]) {
        sdata[tid] = sdata[tid + s];
        sdata[tid + blockDim.x] = sdata[tid + blockDim.x + s];
      }
      // sum of data[i] & data[i]^2
      sdata[tid + 2 * blockDim.x] += sdata[tid + 2 * blockDim.x + s];
      sdata[tid + 3 * blockDim.x] += sdata[tid + 3 * blockDim.x + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    res[blockIdx.x * 4] = sdata[0];
    res[blockIdx.x * 4 + 1] = sdata[blockDim.x];
    res[blockIdx.x * 4 + 2] = sdata[2 * blockDim.x];
    res[blockIdx.x * 4 + 3] = sdata[3 * blockDim.x];
  }
}

//"MAX" reduction for *odata : return max{odata[i]},i
__global__ void get_peak_pos(cufftComplex* odata, float * res, int image_size, const unsigned int N) 
{
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    auto i = blockIdx.x * blockDim.x + tid;
    if (i >= N) return;

    int local_id = i % image_size;

    sdata[tid] = odata[i].x;
    sdata[tid + blockDim.x] = local_id;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) 
    {
        if (tid < s) 
        {
            // find max
            if (sdata[tid + s] > sdata[tid]) 
            {
                sdata[tid] = sdata[tid + s];
                sdata[tid + blockDim.x] = sdata[tid + blockDim.x + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) 
    {
        res[blockIdx.x * 2] = sdata[0];
        res[blockIdx.x * 2 + 1] = sdata[blockDim.x];
    }
}

// CUFFT will enlarge VALUE to N times. Restore it
__global__ void scale(cufftComplex * data, int size, int l2) 
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    data[i].x /= l2;
    data[i].y /= l2;
}

__global__ void clear_image(cufftComplex* data) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i].x = 0;
  data[i].y = 0;
}

__global__ void clear_float(float* data) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  data[i] = 0;
}

__global__ void Complex2float(float* f, cufftComplex* c, int N) 
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  f[i] = c[i].x;
}

__global__ void float2Complex(cufftComplex * c, float * f, int N) 
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    c[i].x = f[i];
    c[i].y = 0;
}

__global__ void do_phase_flip(cufftComplex * filter, Parameters para, int nx, int ny) 
{
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx * ny) return;

    int x = i % nx;
    int y = i / nx;
    //float v = CTF_AST(x, (y + ny / 2) % ny, nx, ny, para.apix, para.dfu, para.dfv, para.dfdiff, para.dfang, para.lambda, para.cs, para.ampconst, 0);
    float v = CTF(x, y, nx, ny, para.apix, para.dfu, para.dfv, para.dfdiff, para.dfang, para.lambda, para.cs, para.ampconst, 0);

    filter[i].x *= v;
    filter[i].y *= v;
}

__global__ void ap2ri(cufftComplex* data, unsigned int N) 
{
  // i <==> global ID
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // ap2ri 将极坐标转换为直角坐标
  float tmp = data[i].x * sinf(data[i].y);
  data[i].x = data[i].x * cosf(data[i].y);
  data[i].y = tmp;
}

__global__ void ri2ap(cufftComplex * data, size_t size) 
{
    // i <==> global ID
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // ri2ap 将直角坐标转换为极坐标
    float tmp = hypotf(data[i].x, data[i].y);
    if (data[i].x == 0 && data[i].y == 0)
        data[i].y = 0;
    else
        data[i].y = atan2(data[i].y, data[i].x);
    data[i].x = tmp;
}
