#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path

def generate_config_files(input_file, output_dir, gpu_values):
    """
    生成配置文件的多个变体
    
    参数:
    input_file: 输入配置文件路径
    output_dir: 输出目录
    part_values: part值的列表
    
    返回:
    生成的文件路径列表
    """
    
    # 读取原始文件内容
    with open(input_file, 'r') as f:
        original_content = f.read()

    #print(original_content)
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件名（不带扩展名）
    input_name = "config" #Path(input_file).stem[:-1]
    #print(input_file)
    input_d = int(input_file.split('fig')[1])
    #print(input_d)
    #input_ext = Path(input_file).suffix
    
    # 生成所有变体文件
    generated_files = []
    for i, gpu in enumerate(gpu_values):
        # 创建修改后的内容
        modified_content = original_content
        
        # 替换part值（所有出现的位置）
        modified_content = re.sub(rf'part{input_d}', f'part{input_d+i+1}', modified_content)

        #print(modified_content)
        
        # 替换GPU_ID值
        modified_content = re.sub(
            r'GPU_ID\s*=\s*\d', 
            f'GPU_ID = {gpu}', 
            modified_content
        )
        
        # 创建文件名
        output_name = f"{input_name}{input_d+i+1}"
        output_file = output_path / output_name
        
        # 保存文件
        with open(output_file, 'w') as f:
            f.write(modified_content)
        
        generated_files.append(str(output_file))
    
    return generated_files

def main():
    parser = argparse.ArgumentParser(description='isSPA_1.1.12 by Zhang Lab at Institute of Biophysics, Chinese Academy of Sciences\nGenerate a batch of configuration files')
    parser.add_argument('input', help='The first configuration file')
    parser.add_argument('-o', metavar='--output', dest='output', default='./', help='The output directory')
    parser.add_argument('--gpus', nargs='+', required=True, help='GPU ID list（e.g., 1 2 3）')
    
    args = parser.parse_args()
    
    print(f"Input: {args.input}")
    print(f"Ouput directory: {args.output}")
    print(f"GPU list: {args.gpus}")
    print("-" * 50)
    
    try:
        # 生成文件
        generated_files = generate_config_files(
            args.input, 
            args.output, 
            args.gpus
        )
        
        print("\nSuccessfully generate the following configuration files:")
        for file_path in generated_files:
            print(f"  - {os.path.basename(file_path)}")
            
    except Exception as e:
        print(f"生成文件时出错: {str(e)}")

if __name__ == "__main__":
    main()