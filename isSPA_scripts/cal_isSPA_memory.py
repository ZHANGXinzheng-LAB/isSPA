import os
import mrcfile
import numpy as np


def cal_isSPA_memory(params, n_g):
    with open(params['angle_step_file'], 'r') as f:
        angles = f.readlines()

    N_angle = len(angles) - 5
    batch_size = np.ceil(N_angle / n_g)
    N_pixel = 512*512*batch_size
    N_pixel2 = 512*512*batch_size

    for file in os.listdir(params['micrographs_dir']):    
        if file.endswith('.mrc'):
            example_file = params['micrographs_dir']+file
            break

    with mrcfile.open(example_file) as f:
        data = f.data

    Lx, Ly = data.shape
    #Lx = 3838
    #Ly = 3710
    binfactor = params['bin']
    pz = params['pixel_size']
    D = np.round(params['d'] * 1.15, 6)

    nx = np.ceil(Lx/(512*binfactor-D/pz))
    ny = np.ceil(Ly/(512*binfactor-D/pz))
    N_pixel3 = 512*512*nx*ny
    N_pixel1 = 512*512*nx*ny
    CCG = N_pixel2*8/1024/1024
    CCG_sum = N_pixel3*8/1024/1024
    CCG_buf = N_pixel2*8/1024/1024
    image = Lx*Ly*4/1024/1024
    padded_image = N_pixel1*8/1024/1024
    padded_templates = N_pixel*8/1024/1024
    ra = batch_size*5000*4/1024/1024
    rb = batch_size*5000*4/1024/1024
    reduction_buf = 2*N_pixel/256*4/1024/1024
    means = batch_size*4/1024/1024
    gpu_memory = (CCG+CCG_sum+CCG_buf+image+padded_image+ra+rb+reduction_buf+means+padded_templates)*1.05

    return np.ceil(gpu_memory)