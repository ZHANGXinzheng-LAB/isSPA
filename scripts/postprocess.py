#!/usr/bin/env python3

import argparse
import numpy as np
import math

def main():
    """通过颗粒的坐标和欧拉角距离将重复颗粒排除"""
    inlst, num, center, euler_thres, scale, apix, mdir, output =  parse_command_line()

    if mdir[-1] == '/':
        mdir = mdir[:-1]


    output = inlst.split('.')[0] + output
    output1 = inlst.split('.')[0] + '_merge.lst'
    with open(inlst, "r") as g:
        inlst_lines = g.readlines()

    # 整理合并后的文件名顺序
    inlst_lines = sorted(inlst_lines, key=lambda line: line.split('\t')[1].split('/')[-1])
    #inlst_lines = sorted(inlst_lines, key=lambda line: line.split('\t')[4].split('=')[-1])
    # 将颗粒信息根据照片分开排序 sort particle information according to micrographs
    particles_list = [[0 for x in range(len(inlst_lines))] for y in range(num)]
    number = [0]*num # 用于统计每张照片中的颗粒数 count the number of particles in each micrograph
    
    name = inlst_lines[0].split('\t')[1] # 照片名称 Micrograph name
    jj = 0 # 用于照片编码
    kk = 0 # 用于统计一张照片内的颗粒数
    for i in inlst_lines:
        if i.split('\t')[1] == name:
            particles_list[jj][kk] = i
            kk += 1
        else:
            number[jj] = kk
            jj += 1
            kk = 0
            name = i.split('\t')[1]
            particles_list[jj][kk] = i
            kk += 1
    number[jj] = kk

    for k in range(jj+1):
        a = [0]*number[k]
        for i in range(number[k]):
            # num = int(inlst_line1[i].split('\t')[0])
            # 提取中心坐标 get center coordinates
            x1 = float(particles_list[k][i].split('\t')[6].split('=')[1].split(',')[0])
            y1 = float(particles_list[k][i].split('\t')[6].split('=')[1].split(',')[1])
            # 提取三个欧拉角 get three Euler angles
            # EMAN对于欧拉角的定义和常用的定义之间差一个负号，不过此处不影响
            phi = float(particles_list[k][i].split('\t')[5].split('=')[1].split(',')[1])*np.pi/180
            theta = float(particles_list[k][i].split('\t')[5].split('=')[1].split(',')[0])*np.pi/180
            psi = float(particles_list[k][i].split('\t')[5].split('=')[1].split(',')[2])*np.pi/180
            # 提取得分 get score
            score1 = float(particles_list[k][i].split('\t')[7].split('=')[1])

            # 检查该颗粒是否被排除
            if a[i]==1:
                continue

            for j in range(i+1, number[k]):
                if a[j] == 1:
                    continue
                x2 = float(particles_list[k][j].split('\t')[6].split('=')[1].split(',')[0])
                y2 = float(particles_list[k][j].split('\t')[6].split('=')[1].split(',')[1])
                dist = np.abs(np.sqrt((x2-x1)**2 + (y2-y1)**2))
                score2 = float(particles_list[k][j].split('\t')[7].split('=')[1])

                if dist < center:
                    phi_o = float(particles_list[k][j].split('\t')[5].split('=')[1].split(',')[1])*np.pi/180
                    theta_o = float(particles_list[k][j].split('\t')[5].split('=')[1].split(',')[0])*np.pi/180
                    psi_o = float(particles_list[k][j].split('\t')[5].split('=')[1].split(',')[2])*np.pi/180
                    # 为了简洁 for simplicity
                    cc = np.cos(theta/2)*np.cos(theta_o/2)
                    ss = np.sin(theta/2)*np.sin(theta_o/2)
                    c1 = np.cos((psi+phi)/2)
                    c2 = np.cos((psi_o+phi_o)/2)
                    C1 = np.cos((psi-phi)/2)
                    C2 = np.cos((psi_o-phi_o)/2)
                    s1 = np.sin((psi+phi)/2)
                    s2 = np.sin((psi_o+phi_o)/2)
                    S1 = np.sin((psi-phi)/2)
                    S2 = np.sin((psi_o-phi_o)/2)
                    # 将欧拉角转换为四元数 convert Euler angles to quaternion
                    r0 = cc*c1*c2 + ss*C1*C2 + ss*S1*S2 + cc*s1*s2
                    r0 = round(r0, 10) # 浮点数误差
                    euler_dist = np.abs(np.arccos(r0)*2*180/np.pi)
                    if euler_dist < euler_thres:
                         #print("score1="+str(score1)+"\t"+"score2="+str(score2))
                        if score1 > score2:
                            a[j] = 1 
                        else:
                            a[i] = 1  
        with open(output1, "a+") as oo:
            for l in range(number[k]):
                if a[l] == 0:        
                    oo.write(particles_list[k][l]) 
    # 将lst文件转换为star文件
    with open(output1, "r") as g:
        inlst_lines = g.readlines()

    #num = 0
    #mag = 50000/apix
    #k = 3

    data = []
    for i in inlst_lines:
        file_path = i.split('\t')[1] # 照片文件路径
        sub_dir = len(file_path.split('/'))
        # 测试是否包含子文件夹
        if sub_dir > 0:
            file_name = file_path.split('/')[-1]
            # 去掉文件名末尾的_bin2
            if scale > 1:
                file_name = file_name.replace(file_name[-9:], '.mrc')
            micrograph_name = mdir + '/' + file_name # 照片文件名称
        else:
            if scale > 1:
                file_path = file_path.replace(file_path[-9:], '.mrc')
            micrograph_name = mdir + '/' + file_path

        
        df = float(i.split('\t')[2].split('=')[1])*10000 # 欠焦值 (angstrom)
        dfdiff = float(i.split('\t')[3].split('=')[1])*10000 # 像散 (angstrom)
        dfu = df - dfdiff
        dfv = df + dfdiff
        dfang = float(i.split('\t')[4].split('=')[1])
        #ox = float(inlst_line2[m+3].split('\t')[5].split('=')[1].split(',')[0])
        #oy = float(inlst_line2[m+3].split('\t')[5].split('=')[1].split(',')[1])
        euler1 = float(i.split('\t')[5].split('=')[1].split(',')[0])
        euler2 = float(i.split('\t')[5].split('=')[1].split(',')[1])
        euler3 = float(i.split('\t')[5].split('=')[1].split(',')[2])
        score = float(i.split('\t')[7].split('=')[1])
        #   m[i-3] = int(inlst_lines[i].split('\t')[0])
        #s5 = "euler="
        #s6 = "center="
        cx = (float(i.split('\t')[6].split('=')[1].split(',')[0]))*scale
        cy = (float(i.split('\t')[6].split('=')[1].split(',')[1]))*scale
        dfu = round(dfu, 10) # 浮点数误差
        dfv = round(dfv, 10)
        # 将EMAN2的Euler角方式转换为RELION的Euler角方式
        euler1 = round(euler1, 4)
        euler2 = math.fmod(euler2-90, 360)
        euler2 = round(euler2, 4)
        euler3 = math.fmod(euler3+90, 360)
        euler3 = round(euler3, 4)
        data.append(str(micrograph_name)+"\t"+str(cx)+"\t"+str(cy)+"\t"+str(dfu)+"\t"+str(dfv)+"\t"+str(dfang)+"\t300.000000\t2.700000\t0.070000\t"+str(apix)+"\t"+str(euler2)+"\t"+str(euler1)+"\t"+str(euler3)+"\n")
    with open(output, "w") as oo:
        oo.write("# RELION; version 3.0-beta-2\n\ndata_\n\nloop_\n_rlnMicrographName #1 \n_rlnCoordinateX #2 \n_rlnCoordinateY #3 \n_rlnDefocusU #4 \n_rlnDefocusV #5 \n_rlnDefocusAngle #6 \n_rlnVoltage #7 \n_rlnSphericalAberration #8 \n_rlnAmplitudeContrast #9 \n_rlnDetectorPixelSize #10 \n_rlnAngleRot #11 \n_rlnAngleTilt #12 \n_rlnAnglePsi #13 \n")
        for j in data:
            oo.write(j)
    

def parse_command_line():
    #提取输入的各个参数
    #usage = "%prog <Detection file> <Number of windows> <Center threshold> <Euler threshold> <Output>"
    parser = argparse.ArgumentParser(description='Remove duplicated particles')
    parser.add_argument('input', metavar='Input_file', help='The .lst file generated by isSPA from target detection')
    parser.add_argument('num', metavar='N', type=int, help='Number of micrographs')
    parser.add_argument('center', metavar='d', type=float, help='The largest distance (pixel) between duplicated points')
    parser.add_argument('euler', metavar='Angle_threshold', type=float, help='The largest angle between two Euler angles')
    parser.add_argument('scale', metavar='Scale', type=int, help='The bining factor used during target detection')
    parser.add_argument('pixel', metavar='Pixel_size', type=float, help='The original pixel size (angstrom) of the original micrograph')
    parser.add_argument('dir', metavar='Micrograph_dir', help='The directory of the original micrographs')
    parser.add_argument('-o', metavar='--output', dest='output', default='_merge.star', help='The output file. Default name: {input}_merge.star')

    args = parser.parse_args()

    inlst = args.input
    num = args.num
    center = args.center
    euler_thres = args.euler
    scale = args.scale
    apix = args.pixel
    mdir = args.dir
    output = args.output

    return inlst, num, center, euler_thres, scale, apix, mdir, output

if __name__== "__main__":
    main()