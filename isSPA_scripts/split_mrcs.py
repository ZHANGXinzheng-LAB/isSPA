#!/usr/bin/env python3

import mrcfile
import numpy as np
import os
import argparse
from tqdm import tqdm
from pathlib import Path
import sys

def split_mrcs_to_individual_files(input_mrcs, output_dir, prefix="particle", start_index=1, digit_pad=4, overwrite=False):
    """
    将MRCS文件拆分为单个MRC文件
    
    参数:
    input_mrcs: 输入的MRCS文件路径
    output_dir: 输出目录
    prefix: 文件名前缀 (默认为"particle")
    start_index: 起始索引 (默认为1)
    digit_pad: 数字填充位数 (默认为4, 如0001)
    overwrite: 是否覆盖已存在的文件 (默认为False)
    
    返回:
    成功保存的颗粒数量
    """
    try:
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 打开MRCS文件
        with mrcfile.mmap(input_mrcs, mode='r', permissive=True) as mrc:
            data = mrc.data
            
            # 验证数据维度
            if data.ndim != 3:
                raise ValueError(f"输入文件应为3D数据 (颗粒, 高度, 宽度)，但当前维度为 {data.ndim}D")
            
            total_particles = data.shape[0]
            
            # 创建进度条
            with tqdm(total=total_particles, desc="提取颗粒") as pbar:
                saved_count = 0
                
                for i in range(total_particles):
                    # 生成输出文件名
                    particle_num = start_index + i
                    particle_str = str(particle_num).zfill(digit_pad)
                    output_file = output_path / f"{prefix}_{particle_str}.mrc"
                    
                    # 检查文件是否已存在
                    if output_file.exists() and not overwrite:
                        pbar.set_postfix_str(f"跳过已存在文件: {output_file.name}")
                        pbar.update(1)
                        continue
                    
                    # 提取单个颗粒
                    particle_data = data[i]
                    
                    # 创建MRC文件
                    with mrcfile.new(output_file, overwrite=overwrite) as new_mrc:
                        # 写入数据
                        new_mrc.set_data(particle_data)
                        
                        # 复制原始头信息 (可选)
                        # new_mrc.header = mrc.header.copy()
                        
                        # 更新关键头信息
                        new_mrc.header.nx = particle_data.shape[1]
                        new_mrc.header.ny = particle_data.shape[0]
                        new_mrc.header.nz = 1
                        new_mrc.update_header_stats()
                    
                    saved_count += 1
                    pbar.set_postfix_str(f"保存: {output_file.name}")
                    pbar.update(1)
            
            return saved_count
            
    except Exception as e:
        print(f"\n处理失败: {str(e)}")
        # 清理可能创建的不完整文件
        if 'i' in locals():
            for j in range(i+1):
                particle_num = start_index + j
                particle_str = str(particle_num).zfill(digit_pad)
                output_file = output_path / f"{prefix}_{particle_str}.mrc"
                if output_file.exists():
                    os.remove(output_file)
        raise

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='MRCS文件拆分工具')
    parser.add_argument('input', help='输入的MRCS文件路径')
    parser.add_argument('output', help='输出目录')
    parser.add_argument('--prefix', default='particle', 
                       help='输出文件前缀 (默认: particle)')
    parser.add_argument('--start', type=int, default=1, 
                       help='起始索引 (默认: 1)')
    parser.add_argument('--digits', type=int, default=4, 
                       help='数字填充位数 (默认: 4)')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的文件')
    
    args = parser.parse_args()
    
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"文件前缀: {args.prefix}")
    print(f"起始索引: {args.start}")
    print(f"数字位数: {args.digits}")
    print(f"覆盖模式: {'是' if args.overwrite else '否'}")
    print("-" * 50)
    
    try:
        # 执行拆分
        saved_count = split_mrcs_to_individual_files(
            input_mrcs=args.input,
            output_dir=args.output,
            prefix=args.prefix,
            start_index=args.start,
            digit_pad=args.digits,
            overwrite=args.overwrite
        )
        
        print(f"\n成功保存 {saved_count} 个颗粒文件到 {args.output}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)