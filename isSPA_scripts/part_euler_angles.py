import numpy as np
import shutil

def part_euler_angles(file, p, head_file):
    with open(file, 'r') as f:
        lines = f.readlines()[3:-2]

    file_name = file.split('/')[-1].split('.txt')[0]
    psi_angles = []
    theta_angles = []
    phi_angles = []

    # 根据需求将欧拉角分组
    if p == 1:
        for i in lines:
            phi = np.float32(i.split('\t')[1]) - 90 # rot
            theta = np.float32(i.split('\t')[2]) # tilt
            psi = np.float32((i.split('\t')[3]).split('\\')[0]) + 90 # psi
            psi_angles.append(psi)
            theta_angles.append(theta)
            phi_angles.append(phi)
        #optics = []
        #for i in phi_angles:
        #    optics.append(1)
        data = np.stack([phi_angles, theta_angles, psi_angles], axis=1)
        shutil.copyfile(head_file, f'{file_name}.star')
        with open(f'{file_name}.star', 'a') as f:
            np.savetxt(f, data, delimiter='\t', fmt='%3.4f')
        return [f'{file_name}.star']
    elif p == 2:
        p_n = round(len(lines) / 2)
        lines1 = lines[:p_n]
        lines2 = lines[p_n:]
        for i in lines1:
            phi = np.float32(i.split('\t')[1]) - 90 # rot
            theta = np.float32(i.split('\t')[2]) # tilt
            psi = np.float32((i.split('\t')[3]).split('\\')[0]) + 90 # psi
            psi_angles.append(psi)
            theta_angles.append(theta)
            phi_angles.append(phi)
        data = np.stack([phi_angles, theta_angles, psi_angles], axis=1)
        shutil.copyfile(head_file, f'{file_name}_part1.star')
        with open(f'{file_name}_part1.star', 'a') as f:
            np.savetxt(f, data, delimiter='\t', fmt='%3.4f')
        psi_angles = []
        theta_angles = []
        phi_angles = []
        for i in lines2:
            phi = np.float32(i.split('\t')[1]) - 90 # rot
            theta = np.float32(i.split('\t')[2]) # tilt
            psi = np.float32((i.split('\t')[3]).split('\\')[0]) + 90 # psi
            psi_angles.append(psi)
            theta_angles.append(theta)
            phi_angles.append(phi)
        data = np.stack([phi_angles, theta_angles, psi_angles], axis=1)
        shutil.copyfile(head_file, f'{file_name}_part2.star')
        with open(f'{file_name}_part2.star', 'a') as f:
            np.savetxt(f, data, delimiter='\t', fmt='%3.4f')
        return [f'{file_name}_part1.star', f'{file_name}_part2.star']
    elif p > 2:
        p_n = round(len(lines)/p)
        files_list = []
        for i in range(p):
            psi_angles = []
            theta_angles = []
            phi_angles = []
            if i == p-1:
                start = i*p_n
                split_lines = lines[start:]
            else:
                start = i*p_n
                end = (i+1)*p_n
                split_lines = lines[start:end]
            for j in split_lines:
                phi = np.float32(j.split('\t')[1]) - 90 # rot
                theta = np.float32(j.split('\t')[2]) # tilt
                psi = np.float32((j.split('\t')[3]).split('\\')[0]) + 90 # psi
                psi_angles.append(psi)
                theta_angles.append(theta)
                phi_angles.append(phi)
            data = np.stack([phi_angles, theta_angles, psi_angles], axis=1)
            # 复制文件头
            shutil.copyfile(head_file, f'{file_name}_part{i+1}.star')
            with open(f'{file_name}_part{i+1}.star', 'a') as f:
                np.savetxt(f, data, delimiter='\t', fmt='%3.4f')
            files_list.append(f'{file_name}_part{i+1}.star')
        return files_list