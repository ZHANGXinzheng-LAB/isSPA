#!/usr/bin/env python3

import os
import re

def modify_star_file_micrograph_names(input_star_file, output_star_file):
    """
    修改STAR文件中的_rlnMicrographName字段
    
    参数:
        input_star_file: 输入STAR文件路径
        output_star_file: 输出STAR文件路径
    """
    
    def extract_falcon_name(original_name):
        """
        从原始微镜图像名称中提取Falcon名称
        
        参数:
            original_name: 原始微镜图像名称
            
        返回:
            提取后的Falcon名称
        """
        # 使用正则表达式提取Falcon名称
        # 匹配模式: 任意字符 + Falcon4_ + 5位数字
        match = re.search(r'(Falcon\d+_\d+)', original_name)
        if match:
            return match.group(1)
        else:
            # 如果没有匹配到，返回原始名称（不含路径和扩展名）
            base_name = os.path.basename(original_name)
            name_without_ext = os.path.splitext(base_name)[0]
            return name_without_ext
    
    def transform_micrograph_name(original_name):
        """
        转换微镜图像名称
        
        参数:
            original_name: 原始微镜图像名称
            
        返回:
            转换后的微镜图像名称
        """
        # 提取Falcon名称
        falcon_name = extract_falcon_name(original_name)
        
        # 构建新的名称
        new_name = f"MotionCorr/job090/Movies/{falcon_name}.mrc"
        
        return new_name
    
    # 读取输入文件并处理
    with open(input_star_file, 'r') as infile, open(output_star_file, 'w') as outfile:
        in_particles_section = False
        micrograph_col_index = -1
        
        for line in infile:
            stripped_line = line.strip()
            
            # 检查是否进入particles部分
            if stripped_line == "data_particles":
                in_particles_section = True
                outfile.write(line)
                continue
            
            # 检查是否离开particles部分
            if stripped_line.startswith("data_") and stripped_line != "data_particles":
                in_particles_section = False
            
            # 如果在particles部分
            if in_particles_section:
                # 检查是否是列定义行
                if stripped_line.startswith("_rlnMicrographName #"):
                    # 提取列索引
                    micrograph_col_index = int(stripped_line.split("#")[1]) - 1
                    outfile.write(line)
                    continue
                
                # 如果是数据行
                if micrograph_col_index != -1 and not stripped_line.startswith("_") and stripped_line:
                    # 分割数据行
                    parts = stripped_line.split()
                    
                    # 确保有足够的列
                    if len(parts) > micrograph_col_index:
                        # 修改微镜图像名称
                        original_name = parts[micrograph_col_index]
                        new_name = transform_micrograph_name(original_name)
                        parts[micrograph_col_index] = new_name
                        
                        # 重新组合行
                        new_line = " ".join(parts) + "\n"
                        outfile.write(new_line)
                    else:
                        outfile.write(line)
                else:
                    outfile.write(line)
            else:
                outfile.write(line)
    
    print(f"处理完成! 输出文件: {output_star_file}")

# 主程序
if __name__ == "__main__":
    input_file = "run_data.star"  # 替换为你的输入STAR文件路径
    output_file = "output_test.star"  # 输出STAR文件路径
    
    modify_star_file_micrograph_names(input_file, output_file)