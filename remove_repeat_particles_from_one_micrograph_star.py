#!/usr/bin/env python3

import numpy as np

def remove_repeat_particles_from_one_micrograph_star(q, number, particles_list, center, euler_thres, scale, spliter, score_threshold):
    a = [0]*number
    for i in range(number):
        # 检查该颗粒是否被排除
        if a[i]==1:
            continue
        # 提取得分 get score
        score1 = float(particles_list[i].split(spliter)[-1].replace('#', ''))
        if score1 <= score_threshold:
            a[i] = 1
            continue
        # 提取中心坐标 get center coordinates
        x1 = float(particles_list[i].split(spliter)[1])
        y1 = float(particles_list[i].split(spliter)[2])
        # 提取三个欧拉角 get three Euler angles
        # EMAN对于欧拉角的定义和常用的定义之间差一个负号，不过此处不影响
        phi = float(particles_list[i].split(spliter)[6])*np.pi/180
        theta = float(particles_list[i].split(spliter)[7])*np.pi/180
        psi = float(particles_list[i].split(spliter)[8])*np.pi/180

        for j in range(i+1, number):
            if a[j] == 1:
                continue
            score2 = float(particles_list[j].split(spliter)[-1].replace('#', ''))
            if score2 <= score_threshold:
                a[j] = 1
                continue
            x2 = float(particles_list[j].split(spliter)[1])
            y2 = float(particles_list[j].split(spliter)[2])
            dist = np.abs(np.sqrt((x2-x1)**2 + (y2-y1)**2))

            if dist <= center:
                phi_o = float(particles_list[j].split(spliter)[6])*np.pi/180
                theta_o = float(particles_list[j].split(spliter)[7])*np.pi/180
                psi_o = float(particles_list[j].split(spliter)[8])*np.pi/180
                '''
                为了简洁 for simplicity
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
                将欧拉角转换为四元数 convert Euler angles to quaternion
                r0 = cc*c1*c2 + ss*C1*C2 + ss*S1*S2 + cc*s1*s2
                '''
                r0 = np.cos(theta/2)*np.cos(theta_o/2) * np.cos((psi+phi)/2) * np.cos((psi_o+phi_o)/2) + np.sin(theta/2)*np.sin(theta_o/2) * np.cos((psi-phi)/2) * np.cos((psi_o-phi_o)/2) + np.sin(theta/2)*np.sin(theta_o/2) * np.sin((psi-phi)/2) * np.sin((psi_o-phi_o)/2) + np.cos(theta/2)*np.cos(theta_o/2) * np.sin((psi+phi)/2) * np.sin((psi_o+phi_o)/2)
                r0 = round(r0, 10) # 浮点数误差
                euler_dist = np.abs((np.arccos(r0)*2*180/np.pi + 180)%360 - 180)
                if euler_dist <= euler_thres:
                     #print("score1="+str(score1)+"\t"+"score2="+str(score2))
                    if score1 > score2:
                        a[j] = 1 
                    else:
                        a[i] = 1
    for l in range(number):
        if a[l] == 0:
            line_list = particles_list[l].split(spliter) 
            # 去掉bin2/
            line_list[0] = line_list[0].replace(f'bin{scale}/', '') 
            # 恢复原图中的坐标
            line_list[1] = str(float(line_list[1]) * scale) 
            line_list[2] = str(float(line_list[2]) * scale)
            new_line = " ".join(line_list)
            q.put(new_line) 