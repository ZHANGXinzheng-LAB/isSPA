#!/usr/bin/env python3

import os
from collections import defaultdict
import argparse

def parse_star_file(star_file):
    """
    解析 RELION STAR 文件，统计每个颗粒图像中的颗粒数量
    
    参数：
    star_file : 输入 STAR 文件路径
    
    返回：
    dict : {颗粒图像路径: 颗粒数量}
    """
    # 初始化统计字典
    image_counts = defaultdict(int)
    
    # 状态跟踪变量
    in_particles_block = False
    in_loop = False
    image_name_col = -1

    with open(star_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 检测数据块开始
            if line.startswith('data_particles'):
                in_particles_block = True
                continue
                
            if not in_particles_block:
                continue
                
            # 检测循环结构开始
            if line.startswith('loop_'):
                in_loop = True
                column_idx = 0
                continue
                
            # 解析列定义
            if in_loop and line.startswith('_'):
                parts = line.split()
                col_name = parts[0]
                
                # 记录 _rlnImageName 列位置
                if col_name == '_rlnImageName':
                    image_name_col = column_idx
                column_idx += 1
                continue
                
            # 处理数据行
            if in_loop and image_name_col != -1 and line and not line.startswith('#'):
                parts = line.split()
                if len(parts) > image_name_col:
                    full_image_name = parts[image_name_col]
                    
                    # 提取实际文件名（格式：序号@路径）
                    if '@' in full_image_name:
                        _, image_path = full_image_name.split('@', 1)
                        image_counts[image_path] += 1

    return image_counts

def main():
    parser = argparse.ArgumentParser(description='RELION STAR 文件颗粒统计工具')
    parser.add_argument('input', help='输入 STAR 文件路径')
    parser.add_argument('-o', '--output', help='输出统计结果文件路径（可选）')
    
    args = parser.parse_args()
    
    # 执行解析
    counts = parse_star_file(args.input)
    
    # 输出结果
    if args.output:
        with open(args.output, 'w') as f:
            f.write("颗粒图像路径\t颗粒数量\n")
            for path, count in sorted(counts.items()):
                f.write(f"{path}\t{count}\n")
        print(f"结果已保存至 {os.path.abspath(args.output)}")
    else:
        print("\n颗粒图像统计结果：")
        for path, count in sorted(counts.items()):
            print(f"• {path}: {count} 颗粒")

if __name__ == "__main__":
    main()