from read_line_format import read_line_format
import numpy as numpy
import multiprocessing as mp
from remove_repeat_particles_from_one_micrograph_star import remove_repeat_particles_from_one_micrograph_star

def postprocess_star(inlst, num, center, euler_thres, scale, output, score_threshold, cpu):
    output = inlst.split('.star')[0] + f'_{score_threshold}' + output
    star_format = read_line_format(inlst)
    if star_format[-1] == 0:
        spliter = " "
    elif star_format[-1] == 1:
        spliter = "\t"
    else:
        spliter = " "
    with open(inlst, 'r') as f:
        lines = f.readlines()
    head_lines = lines[:star_format[-2]]
    data_lines = lines[star_format[-2]+1:] # 有一个附加行 isSPA score
    with open(output, "w") as f:
        f.writelines(head_lines)

    # 整理合并后的文件名顺序
    data_lines = sorted(data_lines, key=lambda line: line.split(spliter)[0].split('/')[-1])
    # 将颗粒信息根据照片分开排序 sort particle information according to micrographs
    particles_list = [[0 for x in range(len(data_lines))] for y in range(num)]
    # 用于统计每张照片中的颗粒数 count the number of particles in each micrograph
    number = [0]*num 

    # 照片名称 Micrograph name
    name = data_lines[0].split(spliter)[0] 
    jj = 0 # 用于照片编码
    kk = 0 # 用于统计一张照片内的颗粒数
    for i in data_lines:
        if i.split(spliter)[0] == name:
            particles_list[jj][kk] = i
            kk += 1
        else:
            number[jj] = kk
            jj += 1
            kk = 0
            name = i.split(spliter)[0]
            particles_list[jj][kk] = i
            kk += 1
    number[jj] = kk

    # 对每一张照片进行并行计算
    q = mp.Manager().Queue()
    if cpu == 0:
        num_cores = int(mp.cpu_count() / 2) # 只用一半CPU
    else:
        if cpu > math.floor(mp.cpu_count() * 0.9):
            print(" There is NOT so many CPU available! Quit.\n")
            return
        else:
            num_cores = cpu
    with mp.Pool(processes=num_cores) as pool:
        pool.starmap(remove_repeat_particles_from_one_micrograph_star, [(q, number[k], particles_list[k], center, euler_thres, scale, spliter, score_threshold) for k in range(jj+1)])
    all_lines_list = []
    while not q.empty():
        all_lines_list.append(q.get())
    with open(output, "a") as oo:
        oo.writelines(all_lines_list)