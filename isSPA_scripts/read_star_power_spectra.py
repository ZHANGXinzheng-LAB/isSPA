#!/usr/bin/env python3

import numpy as np
import re

def read_star_power_spectra(filename):
    """
    读取包含多个data_power块的STAR文件，返回包含所有功率谱数据的NumPy数组列表
    
    参数：
    filename : STAR文件路径
    
    返回：
    list : 包含每个data_power块的NumPy数组的列表，每个数组形状为(n,4)
           列顺序：[_rlnSpectralIndex, _rlnResolution, 
                   _rlnAngstromResolution, _rlnReferenceSpectralPower]
    """
    # 初始化变量
    data_blocks = []
    current_block = None
    columns = []
    col_indices = {}
    
    # 预编译正则表达式
    header_re = re.compile(r'data_(\w+)')
    loop_re = re.compile(r'loop_')
    column_re = re.compile(r'_rln(\w+)\s+#(\d+)')
    data_re = re.compile(r'^\s*\d')

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 检测数据块开始
            if header_re.match(line):
                current_block = header_re.match(line).group(1)
                if current_block == 'power':
                    columns = []
                    col_indices = {}
                    in_loop = False
                    data = []
                continue
                
            # 只处理data_power块
            if current_block != 'power':
                continue
                
            # 检测循环开始
            if loop_re.match(line):
                in_loop = True
                continue
                
            # 解析列定义
            if in_loop and column_re.match(line):
                match = column_re.match(line)
                col_name = match.group(1).lower()
                col_idx = int(match.group(2)) - 1  # 转换为0-based索引
                columns.append((col_idx, {
                    'spectralindex': 0,
                    'resolution': 1,
                    'resolutionsquared': 2,
                    'angstromresolution': 3,
                    'referencespectralpower': 4,
                    'logamplitudesoriginal': 5
                }[col_name]))
                continue
                
            # 数据行处理
            if in_loop and data_re.match(line):
                # 分割数据并转换为浮点数
                values = list(map(float, line.split()))
                
                # 按列顺序重组数据
                sorted_values = [0.0] * 6
                for orig_idx, new_idx in columns:
                    sorted_values[new_idx] = values[orig_idx]
                
                data.append(sorted_values)
                continue
                
            # 数据块结束处理
            if line.startswith('#') or not line:
                if data:
                    # 转换为NumPy数组并存入列表
                    data_blocks.append(np.array(data))
                    data = []
                in_loop = False
    
        # 处理最后一个数据块
        if data:
            data_blocks.append(np.array(data))
    
    print(f"找到 {len(data_blocks)} 个data_power块")
    for i, arr in enumerate(data_blocks[:3]):
        print(f"\n第 {i+1} 个数据块:")
        print("形状:", arr.shape)
        print("前3行数据示例:")
        print(arr[:3])

    return data_blocks
