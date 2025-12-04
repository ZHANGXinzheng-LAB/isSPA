#!/usr/bin/env python3

import argparse
import numpy as np

def generate_angles(d_theta, r):
    """
    合并匹配指定模式的文件
    
    参数:
    pattern: 文件匹配模式 (例如 "Output_delta3_810kd_8A_n3_snr_part*.lst")
    output_file: 合并后的输出文件路径
    delete_original: 是否删除原始文件
    """
    theta_angles = np.linspace(0, 180, int(180/d_theta)+1)
    angles = []
    num = 1
    for i in theta_angles:
        if i == 0.0 or i == 180.0:
            angles.append([num, 0, i, 0])
            num += 1
            continue
        d_phi = d_theta*r/np.sin(i*np.pi/180)
        phi = 0
        while phi < 360:
            angles.append([num, phi, i, 0])
            num += 1
            phi += d_phi
    
    return np.array(angles)

def main():
    parser = argparse.ArgumentParser(description='isSPA搜索角度生成器')
    parser.add_argument('theta_step', metavar='theta', type=float, help='The step of the second Euler angle (polar angle)')
    parser.add_argument('-r', '--ratio', type=float, default=1, help='The ratio of the first Euler angle step to the second Euler angle step')
    #parser.add_argument('psi_step', metavar='psi', type=float, help='The step of the third Euler angle')
    #parser.add_argument('-o', '--output', type=str, default="C1__all.lst", help='输出文件名 ')
    
    args = parser.parse_args()

    d_theta = args.theta_step
    r = args.ratio
    d_phi = np.round(d_theta * r, 3)
    output = f"C1_phi{d_phi}_theta{d_theta}_mirror.txt"
    
    try:
        data = generate_angles(d_theta, r)
        np.savetxt(output, data, fmt='%.2f', delimiter='\t', header=f"Euler generator by Zhang Lab at Institute of Biophysics, Chinese Academy of Sciences\nQuasi-evenly distribution using Penczek's (1994) approach\n phi step: {d_phi}, theta step: {d_theta}, Symmetry: C1", footer=f"Finished!\nTotal sampling angles: {len(data)}")
        print(f"生成完成！总计{len(data)}个Euler角，文件保存在 C1_phi{d_phi}_theta{d_theta}_mirror.txt")
        
    except Exception as e:
        print(f"出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()