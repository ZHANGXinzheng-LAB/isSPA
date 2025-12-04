#!/usr/bin/env python3

import mrcfile
import numpy as np
import argparse
import os
import sys

def validate_voxel_size(mrc1, mrc2, tolerance=1e-3):
    """验证两个MRC文件的体素尺寸是否一致"""
    size1 = mrc1.tolist()  # 处理各向异性和各向同性格式
    size2 = mrc2.tolist()
    
    # 比较XYZ三个方向的体素尺寸
    if not np.allclose(size1, size2, atol=tolerance):
        raise ValueError(f"像素尺寸不匹配\n"
                        f"文件1: {tuple(size1)} Å\n"
                        f"文件2: {tuple(size2)} Å")

def calculate_ratio(mrc_path1, threshold1, mrc_path2, threshold2, check_voxel=True):
    """主计算函数"""
    with mrcfile.open(mrc_path1, mode='r') as mrc1, \
         mrcfile.open(mrc_path2, mode='r') as mrc2:
    
        # 验证像素尺寸
        if check_voxel:
            validate_voxel_size(mrc1.voxel_size, mrc2.voxel_size)

        # 计算掩膜
        mask1 = mrc1.data >= threshold1
        count1 = np.count_nonzero(mask1)
        
        mask2 = mrc2.data >= threshold2
        count2 = np.count_nonzero(mask2)

        # 处理分母为零的情况
        if count2 == 0:
            raise ZeroDivisionError("第二个文件的阈值像素数量为零!请检查阈值或文件！\n")

        ratio = (count1 / count2) * 100
        return ratio, count1, count2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MRC双文件阈值像素比例计算器')
    parser.add_argument('input1', help='第一个MRC文件路径')
    parser.add_argument('threshold1', type=float, help='第一个文件的阈值')
    parser.add_argument('input2', help='第二个MRC文件路径')
    parser.add_argument('threshold2', type=float, help='第二个文件的阈值')
    parser.add_argument('--skip-voxel-check', action='store_true',
                       help='跳过像素尺寸检查')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='显示详细统计信息')
    
    args = parser.parse_args()

    try:
        ratio, count1, count2 = calculate_ratio(
            args.input1, args.threshold1,
            args.input2, args.threshold2,
            check_voxel=not args.skip_voxel_check
        )

        # 输出结果
        print(f"\n计算结果:")
        print(f"文件1 ({os.path.basename(args.input1)})")
        print(f"- 阈值: {args.threshold1}")
        print(f"- 超阈值像素数: {count1:,}")
        
        print(f"\n文件2 ({os.path.basename(args.input2)})")
        print(f"- 阈值: {args.threshold2}")
        print(f"- 超阈值像素数: {count2:,}")
        
        print(f"\n百分比: {ratio:.2f}% (文件1 / 文件2)")

        # 详细模式输出
        if args.verbose:
            with mrcfile.open(args.input1) as mrc1, \
                 mrcfile.open(args.input2) as mrc2:
                
                print("\n详细统计:")
                print(f"数据维度: {mrc1.data.shape}")
                print(f"文件1像素尺寸: {tuple(mrc1.voxel_size)} Å")
                print(f"文件2像素尺寸: {tuple(mrc2.voxel_size)} Å")
                print(f"总像素数: {mrc1.data.size:,}")

    except FileNotFoundError as e:
        print(f"文件错误: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ZeroDivisionError as e:
        print(f"计算错误: {str(e)}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"验证错误: {str(e)}", file=sys.stderr)
        sys.exit(1)
    #except Exception as e:
     #   print(f"未知错误: {str(e)}", file=sys.stderr)
      #  sys.exit(1)