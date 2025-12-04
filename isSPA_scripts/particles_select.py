#!/usr/bin/env python3

import os
import pandas as pd
from collections import defaultdict

def parse_star_file(star_file_path):
    """
    解析STAR文件，提取数据部分
    
    参数:
        star_file_path: STAR文件路径
        
    返回:
        DataFrame: 包含数据的DataFrame
        List: 列名列表
    """
    particles = []
    current_section = None
    columns = []
    in_loop = False
    
    with open(star_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
                
            # 检测数据块开始
            if line.startswith('data_'):
                current_section = line.split('_', 1)[1]
                continue
                
            # 检测循环开始
            if line == 'loop_':
                in_loop = True
                columns = []
                continue
                
            # 解析字段定义
            if line.startswith('_rln') and in_loop:
                # 提取字段名
                field_name = line.split()[0]
                columns.append(field_name)
                continue
                
            # 解析数据行
            if current_section == 'particles' and columns and line:
                data = line.split()
                if len(data) >= len(columns):
                    particle = {}
                    for i, col in enumerate(columns):
                        try:
                            # 尝试转换为浮点数
                            particle[col] = float(data[i])
                        except ValueError:
                            # 无法转换则保留字符串
                            particle[col] = data[i]
                    particles.append(particle)
                # 如果不在循环中，重置列
                elif not in_loop:
                    columns = []
    
    # 创建DataFrame
    df = pd.DataFrame(particles)
    return df, columns

def extract_matching_rows(star_file1, star_file2, output_file):
    """
    从第二个STAR文件中提取与第一个文件匹配的行
    
    参数:
        star_file1: 第一个STAR文件路径（包含rlnImageName列表）
        star_file2: 第二个STAR文件路径（从中提取匹配的行）
        output_file: 输出文件路径
    """
    # 解析第一个STAR文件，提取rlnImageName列表
    print(f"解析第一个STAR文件: {star_file1}")
    df1, cols1 = parse_star_file(star_file1)
    
    if df1.empty or '_rlnImageName' not in df1.columns:
        print("错误: 第一个STAR文件中未找到_rlnImageName列")
        return
    
    image_names = set(df1['_rlnImageName'].values)
    print(f"从第一个文件中提取了 {len(image_names)} 个唯一的rlnImageName")
    
    # 解析第二个STAR文件
    print(f"解析第二个STAR文件: {star_file2}")
    df2, cols2 = parse_star_file(star_file2)
    
    if df2.empty:
        print("错误: 第二个STAR文件中未找到数据")
        return
    
    # 查找匹配的行
    print("查找匹配的行...")
    matched_rows = df2[df2['_rlnImageName'].isin(image_names)]
    
    print(f"找到 {len(matched_rows)} 个匹配的行")
    
    # 写入输出文件
    print(f"写入输出文件: {output_file}")
    with open(output_file, 'w') as f:
        # 写入STAR文件头
        f.write("# version 50001\n\n")
        
        # 写入optics部分（如果有）
        if '_rlnOpticsGroup' in df2.columns:
            f.write("data_optics\n\n")
            f.write("loop_\n")
            f.write("_rlnVoltage #1\n")
            f.write("_rlnImagePixelSize #2\n")
            f.write("_rlnSphericalAberration #3\n")
            f.write("_rlnAmplitudeContrast #4\n")
            f.write("_rlnOpticsGroup #5\n")
            f.write("_rlnImageSize #6\n")
            f.write("_rlnImageDimensionality #7\n")
            f.write("_rlnOpticsGroupName #8\n")
            
            # 获取唯一的optics组
            '''
            optics_groups = df2[['_rlnVoltage', '_rlnImagePixelSize', '_rlnSphericalAberration', 
                                '_rlnAmplitudeContrast', '_rlnOpticsGroup', '_rlnImageSize', 
                                '_rlnImageDimensionality', '_rlnOpticsGroupName']].drop_duplicates()
            
            for _, row in optics_groups.iterrows():
                f.write(f"{row['_rlnVoltage']} {row['_rlnImagePixelSize']} {row['_rlnSphericalAberration']} ")
                f.write(f"{row['_rlnAmplitudeContrast']} {row['_rlnOpticsGroup']} {row['_rlnImageSize']} ")
                f.write(f"{row['_rlnImageDimensionality']} {row['_rlnOpticsGroupName']}\n")
            
            f.write("\n")
            '''
        
        # 写入particles部分
        f.write("data_particles\n\n")
        f.write("loop_\n")
        
        # 写入列定义
        for i, col in enumerate(cols2, 1):
            f.write(f"{col} #{i}\n")
        
        # 写入匹配的数据行
        for _, row in matched_rows.iterrows():
            line = " ".join([str(row[col]) for col in cols2])
            f.write(f"{line}\n")
    
    print("完成!")

# 主程序
if __name__ == "__main__":
    # 配置参数
    star_file1 = "J125_split0.star"  # 第一个STAR文件路径
    star_file2 = "ground_truth_2p7A_particles.star" # 第二个STAR文件路径
    output_file = "matched_output.star" # 输出文件路径
    
    # 提取匹配的行
    extract_matching_rows(star_file1, star_file2, output_file)