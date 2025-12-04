#!/usr/bin/env python3

import os
from collections import defaultdict
import argparse

from eer_to_metadata import eer_to_metadata

def parse_eer_file(directory, stats):
    """
    解析EER文件，统计stats指标
    
    """

    image_counts = defaultdict(float)
    for file in os.listdir(directory):
        if file.endswith('eer'):
            file_path = os.path.join(directory, file)
            metadata = eer_to_metadata(file_path)
            if stats in metadata:
                if stats == "totalDose":
                    image_counts[file] = metadata[stats]
                    #image_counts[file] = str(float(metadata[stats])/(float(metadata['sensorPixelSize.width'])*1e10)**2)
                else:
                    image_counts[file] = metadata[stats]
            else:
                print(f"警告: 在文件 {file} 中未找到统计指标 '{stats}'")
                exit();
    return image_counts

def main():
    parser = argparse.ArgumentParser(description='EER文件统计工具')
    parser.add_argument('input', help='输入EER文件路径')
    #parser.add_argument('stats', help='统计项目1')
    parser.add_argument('-s', '--stats', default='totalDose', help='统计项目(可选)，默认为总剂量')
    parser.add_argument('-o', '--output', help='输出统计结果文件路径（可选）')
    
    args = parser.parse_args()
    
    # 执行解析
    result = parse_eer_file(args.input, args.stats)
    
    # 输出结果
    if args.output:
        with open(args.output, 'w') as f:
            f.write("图像路径\t总剂量（e/pixel^2）\n")
            for path, count in sorted(result.items()):
                f.write(f"{path}\t{count}\n")
        print(f"结果已保存至 {os.path.abspath(args.output)}")
    else:
        print("\n图像统计结果：")
        for path, count in sorted(result.items()):
            print(f"{path}，剂量（e/pixel^2）: {count} ")

if __name__ == "__main__":
    main()