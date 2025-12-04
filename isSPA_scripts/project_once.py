#!/usr/bin/env python3
"""
增强版 RELION 投影脚本 - 支持参数化操作和批量处理
"""
import argparse
import glob
import os
import subprocess
import sys
import logging
from pathlib import Path
from part_euler_angles import part_euler_angles

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("relion_projector")

def run_relion_project(template: Path, input_star: Path, output_prefix: str):
    """执行 relion_project 命令"""
    # 构建输出文件名（保留输入文件基本名）
    input_basename = input_star.stem
    output_file = f"{output_prefix}{input_basename}"
    
    # 构建命令
    cmd = [
        "relion_project",
        "--i", str(template),
        "--ang", str(input_star),
        "--o", output_file
    ]
    
    print(f"Projection: {input_star.name} →  {output_file}") 
    logger.debug("执行命令: " + " ".join(cmd))
    
    try:
        # 执行命令并实时输出日志
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"处理失败: {input_star.name}\n错误: {e.output}")
        return False
    except FileNotFoundError:
        logger.error("未找到 relion_project 命令，请确保 RELION 已正确安装")
        sys.exit(1)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="isSPA_1.1.12 by Zhang Lab at Institute of Biophysics, Chinese Academy of Sciences\nGenerate a batch of projections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument("template", type=Path, help="3D Template (e.g., templates/810kd_masked.mrc)")
    
    # 可选参数
    parser.add_argument("input", type=str, help="Angle file (e.g., C1_delta2_mirror.txt)")
    parser.add_argument("-p", "--part", type=int, default=1, help="No. of partition")
    parser.add_argument("--head", type=str, default='relion_projection_head.star', help="RELION file head")
    parser.add_argument("-o", "--output", type=str, default="projections_", help="Output file prefix")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    
    args = parser.parse_args()
    
    # 设置调试模式
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("调试模式已启用")
    
    # 验证模板文件存在
    if not Path(args.input).exists():
        logger.error(f"角度文件不存在: {args.input}")
        sys.exit(1)

    # 验证模板文件存在
    if not args.template.exists():
        logger.error(f"模板文件不存在: {args.template}")
        sys.exit(1)
    
    files_list = part_euler_angles(args.input, args.part, args.head)

    print(f"Angle file: {args.input}\nPartition files: {files_list}")
    print(f"Template: {args.template}")
    print(f"Prefix: {args.output}")
    
    # 处理所有文件
    success_count = 0
    for input_file in sorted(files_list):
        input_path = Path(input_file)
        if run_relion_project(args.template, input_path, args.output):
            success_count += 1

            '''
    for i in files_list:
        with open(i, 'r') as f:
            n_lines = len(f.readlines())
        logger.info(f"文件: {args.output}")
        '''
    
    # 输出总结
    print(f"Projection finished! Success: {success_count}/{len(files_list)}")
    if success_count < len(files_list):
        sys.exit(1)

if __name__ == "__main__":
    main()