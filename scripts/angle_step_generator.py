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
    theta_angles = np.append(np.arange(0,180,d_theta), 180)
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

def write_star(angles, ipart, d_phi, d_theta):
    ofile = f"phi{d_phi}_theta{d_theta}_part{ipart}.star"
    out = open(ofile, 'w')
    out.write(f"#Euler generator by Zhang Lab at Institute of Biophysics, Chinese Academy of Sciences\n")
    out.write(f"#Quasi-evenly distribution using Penczek's (1994) approach\n")
    out.write(f"# phi step: {d_phi}, theta step: {d_theta}, Symmetry: C1\n")
    out.write(f"#Total sampling angles: {len(angles)}\n\n")
    out.write(f"data_optics\n\nloop_\n_pyColumn1 #1\n_rlnMicrographPixelSize #2\n")
    out.write(f"_rlnVoltage #3\n_rlnSphericalAberration #4\nnoFloat 1 300 2.7\n\n")
    out.write(f"data_particles\n\nloop_\n")
    out.write(f"_rlnAngleRot #1\n_rlnAngleTilt #2\n_rlnAnglePsi #3\n")
    _, rot, tilt, psi = angles.T
    for x,y,z in zip(rot, tilt, psi): 
         out.write(f"{x-90:>8.3f} {y:>8.3f} {z+90:>8.3f}\n")
    out.close()
    print(f'已生成{len(angles)}个Euler角，保存在{ofile}')
    

def main():
    parser = argparse.ArgumentParser(description='isSPA搜索角度生成器')
    parser.add_argument('-d', '--delta', type=float, default=3, help='The step of the second Euler angle (polar angle)')
    parser.add_argument('-N', type=int, default=1, help='Number of files to split')
    #parser.add_argument('psi_step', metavar='psi', type=float, help='The step of the third Euler angle')
    #parser.add_argument('-o', '--output', type=str, default="C1__all.lst", help='输出文件名 ')
    
    args = parser.parse_args()

    d_theta = args.delta
    r = 1 # The ratio of the first Euler angle step to the second Euler angle step
    d_phi = np.round(d_theta * r, 3)
    angles = generate_angles(d_theta, r)
    for i,v in enumerate(np.array_split(angles, args.N)): write_star(v, i, d_phi, d_theta)    
    print(f"生成完成！总计{len(angles)}个Euler角")
        

if __name__ == "__main__":
    main()