import numpy as np
import math
import multiprocessing as mp
from remove_repeat_particles_from_one_micrograph import remove_repeat_particles_from_one_micrograph


def postprocess_lst(inlst, num, center, euler_thres, scale, apix, mdir, output, score_threshold, cpu):
    output = inlst.split('.lst')[0] + f'_{score_threshold}' + output
    output1 = inlst.split('.lst')[0] + f'_{score_threshold}' + '_merge.lst'
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
            # 当照片数超过指定数量时，跳过多余的照片
            if jj == num-1:
                break
            number[jj] = kk
            jj += 1
            kk = 0
            name = i.split('\t')[1]
            particles_list[jj][kk] = i
            kk += 1
    number[jj] = kk

    # 对每一张照片进行并行计算
    q = mp.Manager().Queue()
    if cpu == 0:
        num_cores = int(mp.cpu_count() / 2)
    else:
        if cpu > math.floor(mp.cpu_count() * 0.9):
            print(" There are Not so many CPU available! Quit.\n")
            return
        else:
            num_cores = cpu
    with mp.Pool(processes=num_cores) as pool:
        pool.starmap(remove_repeat_particles_from_one_micrograph, [(q, number[k], particles_list[k], center, euler_thres, score_threshold) for k in range(jj+1)])
    all_lines_list = []
    while not q.empty():
        all_lines_list.append(q.get())
    with open(output1, "w") as oo:
        oo.writelines(all_lines_list)
                    
    ### 将lst文件转换为star文件
    with open(output1, "r") as g:
        inlst_lines = g.readlines()

    data = []
    for i in inlst_lines:
        file_path = i.split('\t')[1] # 照片文件路径
        sub_dir = len(file_path.split('/'))
        # 测试是否包含子文件夹
        if sub_dir > 0:
            if scale != 1.0:
                ori_name = file_path.split('_bin')[0]
                micrograph_name = mdir + '/' + ori_name.split('/')[-1] + '.mrc' # 照片文件名称
            else:
                micrograph_name = mdir + '/' + file_path.split('/')[-1]
        else:
            if scale != 1.0:
                ori_name = file_path.split('_bin')[0]
                micrograph_name = mdir + '/' + ori_name + '.mrc'
            else:    
                micrograph_name = mdir + '/' + file_path
        #if(os.path.exists(output)):
        #      oo=open(output,"a")   
        
        df = float(i.split('\t')[2].split('=')[1])*10000 # 欠焦值 (angstrom)
        dfdiff = float(i.split('\t')[3].split('=')[1])*10000 # 像散 (angstrom)
        dfu = df - dfdiff # 和欠焦角度的定义一致
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
        cx = np.round((float(i.split('\t')[6].split('=')[1].split(',')[0]))*scale, 6)
        cy = np.round((float(i.split('\t')[6].split('=')[1].split(',')[1]))*scale, 6)
        dfu = round(dfu, 10) # 浮点数误差
        dfv = round(dfv, 10)
        #dfang = math.fmod(dfang-90, 360)
        # 将EMAN2的Euler角方式转换为RELION的Euler角方式
        euler1 = round(euler1, 4)
        euler2 = math.fmod(euler2-90, 360)
        euler2 = round(euler2, 4)
        euler3 = math.fmod(euler3+90, 360)
        euler3 = round(euler3, 4)
        data.append(str(micrograph_name)+"\t"+str(cx)+"\t"+str(cy)+"\t"+str(dfu)+"\t"+str(dfv)+"\t"+str(dfang)+"\t300.000000\t2.700000\t0.070000\t"+str(apix)+"\t"+str(euler2)+"\t"+str(euler1)+"\t"+str(euler3)+"\t"+str(score)+"\t10000\n")
    with open(output, "w") as oo:
        oo.write("# RELION; version 3.0-beta-2\n\ndata_\n\nloop_\n_rlnMicrographName #1 \n_rlnCoordinateX #2 \n_rlnCoordinateY #3 \n_rlnDefocusU #4 \n_rlnDefocusV #5 \n_rlnDefocusAngle #6 \n_rlnVoltage #7 \n_rlnSphericalAberration #8 \n_rlnAmplitudeContrast #9 \n_rlnDetectorPixelSize #10 \n_rlnAngleRot #11 \n_rlnAngleTilt #12 \n_rlnAnglePsi #13 \n_rlnAutopickFigureOfMerit #14 \n_rlnMagnification #15 \n")
        for j in data:
            oo.write(j)