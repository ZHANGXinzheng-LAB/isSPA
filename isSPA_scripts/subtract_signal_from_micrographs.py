#!/usr/bin/env python3

import mrcfile
import numpy as np
import os
import argparse
import time

from run_relion_project import run_relion_project
from read_line_format import read_line_format
from run_relion_reconstruct import run_relion_reconstruct
from replace_and_copy import replace_and_copy

def subtract_signal(dir_path, n, proj_file, particles_file, mpi=None, ctf=False):
    begin = time.time()

    if dir_path[-1] == '/':
        dir_path = dir_path[:-1]

    # 记录颗粒照片
    micrographs = []
    for file in os.listdir(dir_path):
        # Check only text files
        if file.endswith('.mrcs'):
            micrographs.append(file)
    micrographs.sort()
    micrographs_subset = micrographs[:n]

    # 记录颗粒数量
    num = [0] 
    num_c = 0
    data = []
    for i in micrographs_subset:
        with mrcfile.open(f'{dir_path}/{i}') as f:
            num_c += f.header['nz']
            num.append(num_c)
            data_c = f.data
            if data_c.ndim == 2:
                data_c = np.expand_dims(data_c, axis=0)
            data.append(data_c)
    data = np.concatenate(data, axis=0)
    star_format = read_line_format(particles_file)
    # 生成投影角度文件
    with open(particles_file) as f:
        lines = f.readlines()
    lines_subset = lines[:star_format[-2]+num[-1]]
    particles_filename = particles_file.split('.star')[0]
    subset_particles_file = particles_filename + f'_subset{n}.star'
    with open(subset_particles_file, 'w') as f:
        f.writelines(lines_subset)

    #proj_name = "projections_" + particles_filename + f"_subset{n}"
    #print("开始投影......")
    # 生成投影
    #if run_relion_project(input_map, subset_particles_file, proj_name, ctf=True):

    print("计算噪声......")

    #proj_file = proj_name + ".mrcs"
    # 读取投影文件
    with mrcfile.mmap(proj_file) as f:
        projections = f.data[:num[-1]]
        head = f.header
        voxel_size = f.voxel_size
    
    a = data - projections
    a_ft = np.fft.fftn(a, norm='forward', axes=(1,2))
    a_ft = np.fft.fftshift(a_ft, axes=(1,2))
    a_ft = np.abs(a_ft)**2

    output = f"{particles_filename}_noise_subset{n}"
    with mrcfile.new(f"{output}.mrcs", overwrite=True) as dst_mrc:
        dst_mrc.set_data(a_ft.astype(np.float32))
        
        # 复制关键头信息（可选调整）
        dst_mrc.header.cella = head.cella  # 单位晶胞尺寸
        dst_mrc.header.cellb = head.cellb
        dst_mrc.header.mapc = head.mapc    # 列方向
        dst_mrc.header.mapr = head.mapr
        dst_mrc.header.maps = head.maps
        
        # 调整体素尺寸（假设原文件 Z 轴为切片方向）
        dst_mrc.voxel_size = voxel_size
        
        # 更新数据统计信息（自动计算）
        dst_mrc.update_header_from_data()

    if replace_and_copy(
        src_path=subset_particles_file,
        dst_path=f"{output}.star",
        old_str=particles_filename, 
        new_str=output):
        print("开始重构......")
        if run_relion_reconstruct(f"{output}.star", f"reconstruct_{output}.mrc", mpi, ctf):
            print("RELION 重构完成！")

    end = time.time()
    elapsed = end-begin
    print("运行时间: %.3f s" % (elapsed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='从照片中减去信号')
    parser.add_argument('dir_path', help='颗粒照片所在文件夹')
    parser.add_argument('num', type=int, help='照片数量')
    parser.add_argument('proj', help='投影照片文件 (.mrcs)')
    parser.add_argument('particles', help='颗粒文件 (.star)')
    parser.add_argument("--mpi", type=int, default=None,
                       help="mpi数量")
    parser.add_argument('--ctf', action='store_true', help="重构时对CTF进行修正")

    args = parser.parse_args()
    subtract_signal(args.dir_path, args.num, args.proj, args.particles, args.mpi, args.ctf)