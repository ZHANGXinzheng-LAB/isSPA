#!/usr/bin/env python3

import os
import re
import argparse

def process_fsc_data(input_star_file, output_star_file):
    """
    处理STAR文件中的FSC数据，计算平方值并保存到新文件
    
    参数:
        input_star_file: 输入STAR文件路径
        output_star_file: 输出STAR文件路径
    """
    
    # 读取输入文件
    with open(input_star_file, 'r') as infile:
        lines = infile.readlines()
    
    # 处理FSC数据
    in_fsc_section = False
    in_loop = False
    fsc_col_index = -1
    processed_lines = []
    column_names = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # 检查是否进入data_fsc部分
        if stripped_line == "data_fsc":
            in_fsc_section = True
            processed_lines.append(line)
            continue
        
        # 检查是否离开data_fsc部分
        if stripped_line.startswith("data_") and stripped_line != "data_fsc":
            in_fsc_section = False
        
        # 如果在data_fsc部分
        if in_fsc_section:
            # 检查是否是循环开始
            if stripped_line == "loop_":
                in_loop = True
                processed_lines.append(line)
                continue
            
            # 检查是否是列定义
            if in_loop and stripped_line.startswith("_rln"):
                # 提取列名
                col_name = stripped_line.split()[0]
                column_names.append(col_name)
                
                # 记录_rlnFourierShellCorrelation的列索引
                if col_name == "_rlnFourierShellCorrelation":
                    fsc_col_index = len(column_names) - 1
                
                processed_lines.append(line)
                continue
            
            # 如果是数据行
            if in_loop and fsc_col_index != -1 and stripped_line and not stripped_line.startswith("_"):
                # 分割数据行
                parts = stripped_line.split()
                
                # 确保有足够的列
                if len(parts) > fsc_col_index:
                    # 获取FSC值并计算平方
                    try:
                        fsc_value = float(parts[fsc_col_index])
                        fsc_squared = fsc_value ** 2 / (2 - fsc_value**2)
                        
                        # 添加新列到行尾
                        new_line = line.strip() + f" {fsc_squared:.6f}\n"
                        processed_lines.append(new_line)
                    except ValueError:
                        # 如果无法转换为浮点数，保留原行
                        processed_lines.append(line)
                else:
                    processed_lines.append(line)
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    # 添加新列定义
    # 找到data_fsc部分中loop_后的位置插入新列定义
    for i, line in enumerate(processed_lines):
        if line.strip() == "loop_":
            # 找到loop_后的最后一个列定义
            j = i + 1
            while j < len(processed_lines) and processed_lines[j].strip().startswith("_rln"):
                j += 1
            
            # 在最后一个列定义后插入新列定义
            new_col_index = len(column_names) + 1
            new_col_def = f"_rlnFourierShellCorrelationCorrected #{new_col_index}\n"
            processed_lines.insert(j, new_col_def)
            break
    
    # 写入输出文件
    with open(output_star_file, 'w') as outfile:
        outfile.writelines(processed_lines)
    
    print(f"处理完成! 输出文件: {output_star_file}")


# 主程序
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This program convert FSC to FSC_ref')
    parser.add_argument('input', metavar='Input_star', help='The input FSC file')
    parser.add_argument('-o', metavar='--output', dest='output', default='postprocess_convert.star', help='The output file. Default: postprocess_convert.star')

    args = parser.parse_args()

    input_file = args.input  
    output_file = args.output  # 输出STAR文件路径
    
    # 使用简单版本或健壮版本
    process_fsc_data(input_file, output_file)
    #process_fsc_data_robust(input_file, output_file)