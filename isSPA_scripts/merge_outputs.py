#!/usr/bin/env python3

import os
import glob
import argparse
import sys
import fnmatch

def merge_part_files(pattern, output_file, delete_original=False):
    """
    合并匹配指定模式的文件
    
    参数:
    pattern: 文件匹配模式 (例如 "Output_delta3_810kd_8A_n3_snr_part*.lst")
    output_file: 合并后的输出文件路径
    delete_original: 是否删除原始文件
    """
    found_files = []
    current_directory = os.getcwd()  # Get the current working directory
    for item in os.listdir(current_directory):
        full_path = os.path.join(current_directory, item)
        if os.path.isfile(full_path) and fnmatch.fnmatch(item, f'{pattern}_part[0-9]*_*_merge.lst'):
            found_files.append(full_path)

    # 获取匹配的文件列表
    #files = sorted(glob.glob(pattern))
    
    if not found_files:
        print(f"Error: can NOT find any file matching {pattern}")
        return 0
    
    print(f"Found {len(found_files)} matching files:")
    for file in found_files:
        print(f"  - {file}")
    
    total_lines = 0
    
    # 打开输出文件
    with open(output_file, 'w') as out_f:
        # 逐个处理输入文件
        for file_path in found_files:
            file_lines = 0
            
            # 读取并写入内容
            with open(file_path, 'r') as in_f:
                for line in in_f:
                    out_f.write(line)
                    file_lines += 1
            
            print(f"Merging {file_path}: {file_lines} lines")
            total_lines += file_lines
            
            # 删除原始文件
            if delete_original:
                os.remove(file_path)
                print(f"已删除原始文件: {file_path}")
    
    return total_lines

def main():
    parser = argparse.ArgumentParser(description='isSPA_1.1.12 by Zhang Lab at Institute of Biophysics, Chinese Academy of Sciences\nMerge a batch of output files')
    parser.add_argument('input', type=str, help='File name format (e.g., Output_delta3_810kd_8A_n3_snr_part*_6.6_merge.lst)')
    parser.add_argument('-o', '--output', type=str, default="_all.lst", help='Output file (e.g., Output_delta3_810kd_8A_n3_snr_all.lst)')
    parser.add_argument('--delete', action='store_true', help='delete original output files')
    
    args = parser.parse_args()

    name = args.input.split('_part')[0]
    #ext = args.input.split('_part')[0]
    output = name + args.output
    
    print(f"Pattern: {name}")
    #print(f"Extension: {name}")
    print(f"Output: {output}")
    print(f"Delete original files?: {'No' if not args.delete else 'Yes'}")
    print("-" * 50)
    
    try:
        # 执行合并
        total_lines = merge_part_files(
            name,
            output,
            delete_original=args.delete
        )
        
        print(f"\nMerge completes! Total lines: {total_lines}")
        print(f"Merged file: {os.path.abspath(output)}")
        
    except Exception as e:
        print(f"合并过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()