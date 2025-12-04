#!/usr/bin/env python3

import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CTF_file", required=True)
    parser.add_argument("--pixel_size", type=float, required=True)
    parser.add_argument("--voltage", type=float, required=True)
    parser.add_argument("--angle_step", type=float, required=True)
    parser.add_argument("--window_size", type=int, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    
    args = parser.parse_args()
    
    print(f"启动参数处理:")
    print(f"CTF文件: {args.CTF_file}")
    print(f"像素尺寸: {args.pixel_size} Å")
    print(f"加速电压: {args.voltage} kV")
    print(f"角度步长: {args.angle_step} 度")
    print(f"窗口尺寸: {args.window_size} px")
    print(f"处理图像范围: {args.first}-{args.last}")
    
    script_path = os.path.join(os.path.dirname(__file__), "preprocess.py")
    cmd = [
            "python3", script_path, args.CTF_file, 
            "--i", input_star,
            "--ang", ang_star,
            "--o", output_dir
        ]
    
    try:
        # 执行命令并等待完成
        result = subprocess.run(
            cmd,
            check=True,  # 如果返回非零状态码则抛出异常
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 打印输出日志
        print("relion_project 输出:")
        print(result.stdout)
    
    print("所有图像处理完成")

if __name__ == "__main__":
    main()