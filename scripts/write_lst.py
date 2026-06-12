#!/usr/bin/env python3

import math

def write_lst(lines, star_format, lst, v, spliter, fnum, bin_factor):
    """根据star格式选取相应的物理量，并写入lst文件"""
    meta_lines = lines[star_format[-2]-star_format[v]:star_format[-2]]
    #print(meta_lines)
    for j in meta_lines:
        #print(j.split())
        if j.split()[0] == '_rlnMicrographName':
            filename_n = int(j.split()[-1].split('#')[-1])-1
        elif j.split()[0] == '_rlnDefocusU':
            dfu_n = int(j.split()[-1].split('#')[-1])-1
        elif j.split()[0] == '_rlnDefocusV':
            dfv_n = int(j.split()[-1].split('#')[-1])-1
        elif j.split()[0] == '_rlnDefocusAngle':
            dfang_n = int(j.split()[-1].split('#')[-1])-1
    with open(lst, 'w') as g:
        g.write('#LST\n')
        for i in lines[star_format[-2]:]:
            if i.strip():
                new_line = " ".join(i.split(spliter))
                micrograph_name = new_line.split()[filename_n]
                file_name = micrograph_name.split('/')[-1]
                if file_name in micrograph_name:
                    file_path = micrograph_name.replace(file_name, '')
                if bin_factor > 1:
                    file_path = file_path + f'bin{bin_factor}/'
                    file_name = file_name.replace('.mrc', f'_bin{bin_factor}.mrc')
                #print(new_line.split())
                dfu = float(new_line.split()[dfu_n])/1e4
                dfv = float(new_line.split()[dfv_n])/1e4
                dfang = float(new_line.split()[dfang_n])
                defocus = (dfu + dfv)/2
                dfdiff = math.fabs(dfu-dfv)/2 # dfu can be larger or smaller than dfv
                if dfu > dfv:
                    dfang = math.fmod(dfang + 360 + 90, 360) # change the angle from the larger defocus to the smaller defocus
                g.write(f'{fnum}\t{file_path}{file_name}\tdefocus=%.11f\tdfdiff=%.10f\tdfang=%.6f\n' % (defocus, dfdiff, dfang))
                