def cal_angles(d_theta, r):
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
            #angles.append([num, 0, i, 0])
            num += 1
            continue
        d_phi = d_theta*r/np.sin(i*np.pi/180)
        phi = 0
        while phi < 360:
            #angles.append([num, phi, i, 0])
            num += 1
            phi += d_phi
    
    return num