import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def apply_lowpass_filter_2d(image, cutoff_frequency, filter_type='gaussian'):
    """
    在傅里叶空间对二维数组应用低通滤波器
    
    参数:
        image: 2D numpy数组，输入图像
        cutoff_frequency: 截止频率，值在0到1之间，表示相对于最大频率的比例
        filter_type: 滤波器类型，'gaussian'（高斯）或 'ideal'（理想）
    
    返回:
        filtered_image: 滤波后的2D numpy数组
    """
    # 检查输入参数
    if cutoff_frequency <= 0 or cutoff_frequency > 1:
        raise ValueError("截止频率必须在(0, 1]范围内")
    
    # 计算傅里叶变换
    fft_image = fft2(image)
    fft_shifted = fftshift(fft_image)
    
    # 获取图像尺寸
    rows, cols = image.shape
    
    # 创建频率网格
    u = np.arange(-cols//2, cols//2) / (cols/2)
    v = np.arange(-rows//2, rows//2) / (rows/2)
    U, V = np.meshgrid(u, v)
    
    # 计算到频率中心的距离
    D = np.sqrt(U**2 + V**2)
    
    # 创建滤波器
    if filter_type == 'gaussian':
        # 高斯低通滤波器
        H = np.exp(-(D**2) / (2 * (cutoff_frequency**2)))
    elif filter_type == 'ideal':
        # 理想低通滤波器
        H = np.zeros_like(D)
        H[D <= cutoff_frequency] = 1
    else:
        raise ValueError("滤波器类型必须是'gaussian'或'ideal'")
    
    # 应用滤波器
    filtered_fft = fft_shifted * H
    
    # 逆傅里叶变换
    filtered_fft_ishifted = ifftshift(filtered_fft)
    filtered_image = np.real(ifft2(filtered_fft_ishifted))
    
    return filtered_image