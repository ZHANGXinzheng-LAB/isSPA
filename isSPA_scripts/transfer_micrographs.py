#!/usr/bin/env python3
"""
从STAR文件中提取微镜图像文件名并通过SCP传输到远程服务器
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def parse_star_file(star_file_path):
    """
    解析STAR文件，提取所有的MicrographName
    """
    micrographs = []
    
    try:
        with open(star_file_path, 'r') as f:
            lines = f.readlines()
        
        # 找到数据部分开始的位置
        data_start = False
        header_found = False
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 检测数据表开始
            if line.startswith('loop_'):
                data_start = True
                continue
            
            # 在数据表中查找MicrographName列
            if data_start and line.startswith('_rlnMicrographName'):
                header_found = True
                continue
            
            # 提取数据行
            if data_start and header_found:
                if line and not line.startswith('_'):
                    parts = line.split()
                    if parts:
                        micrograph_path = parts[0]
                        micrographs.append(micrograph_path)
    
    except Exception as e:
        print(f"解析STAR文件时出错: {e}")
        return []
    
    return micrographs

def scp_transfer_files(micrograph_paths, remote_server, remote_path, local_base_path=""):
    """
    通过SCP传输文件到远程服务器
    """
    transferred = []
    failed = []
    
    for micrograph_path in micrograph_paths:
        # 构建完整的本地文件路径
        if local_base_path:
            local_file = os.path.join(local_base_path, micrograph_path)
        else:
            local_file = micrograph_path
        
        # 检查文件是否存在
        if not os.path.exists(local_file):
            print(f"警告: 文件不存在 - {local_file}")
            failed.append(micrograph_path)
            continue
        
        # 构建远程路径
        filename = os.path.basename(micrograph_path)
        remote_file = f"{remote_server}:{remote_path}/{filename}"
        
        try:
            # 执行SCP命令
            print(f"传输: {local_file} -> {remote_file}")
            result = subprocess.run(
                ["scp", "-P3395", local_file, remote_file],
                check=True,
                capture_output=True,
                text=True
            )
            transferred.append(micrograph_path)
            print(f"成功: {micrograph_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"传输失败: {micrograph_path}")
            print(f"错误: {e.stderr}")
            failed.append(micrograph_path)
    
    return transferred, failed

def main():
    parser = argparse.ArgumentParser(description='从STAR文件传输微镜图像到远程服务器')
    parser.add_argument('star_file', help='STAR文件路径')
    parser.add_argument('remote_server', help='远程服务器 (user@hostname)')
    parser.add_argument('remote_path', help='远程服务器上的目标路径')
    parser.add_argument('--local-base', '-l', default='', 
                       help='本地文件的基础路径 (如果STAR中的路径是相对路径)')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='只显示将要传输的文件，不实际执行')
    
    args = parser.parse_args()
    
    # 解析STAR文件
    print(f"解析STAR文件: {args.star_file}")
    micrographs = parse_star_file(args.star_file)
    
    if not micrographs:
        print("未找到任何微镜图像文件")
        return 1
    
    print(f"找到 {len(micrographs)} 个微镜图像文件")
    
    # 显示文件列表
    for i, micrograph in enumerate(micrographs, 1):
        print(f"{i:3d}. {micrograph}")
    
    if args.dry_run:
        print("\nDry-run模式: 不实际传输文件")
        return 0
    
    # 确认传输
    response = input(f"\n是否开始传输 {len(micrographs)} 个文件到 {args.remote_server}? (y/N): ")
    if response.lower() != 'y':
        print("取消传输")
        return 0
    
    # 执行传输
    print("\n开始传输文件...")
    transferred, failed = scp_transfer_files(
        micrographs, 
        args.remote_server, 
        args.remote_path,
        args.local_base
    )
    
    # 输出结果
    print(f"\n传输完成!")
    print(f"成功: {len(transferred)} 个文件")
    print(f"失败: {len(failed)} 个文件")
    
    if failed:
        print("\n失败的文件:")
        for f in failed:
            print(f"  - {f}")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())