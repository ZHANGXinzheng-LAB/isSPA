#!/usr/bin/env python3
from collections import defaultdict
import os, argparse
import mrcfile

def count_mrcs(dir_path):
    #dir_path = './Polish/job035/convert_micrographs/Movies'
    image_counts = defaultdict(int)
    # 记录颗粒照片
    micrographs = []
    for file in os.listdir(dir_path):
        # Check only text files
        if file.endswith('.mrcs'):
            micrographs.append(file)
    micrographs.sort()
    micrographs_subset = micrographs

    # 记录颗粒数量
    num = [0]
    num_c = 0
    for i in micrographs_subset:
        with mrcfile.open(f'{dir_path}/{i}') as f:
            image_counts[f'{dir_path}/{i}'] = f.header['nz']

    return image_counts

def main():
    parser = argparse.ArgumentParser(description='.mrcs 文件颗粒统计工具')
    parser.add_argument('input', help='输入 .mrcs 文件路径')
    parser.add_argument('-o', '--output', help='输出统计结果文件路径（可选）')
    
    args = parser.parse_args()
    
    # 执行解析
    counts = count_mrcs(args.input)
    
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