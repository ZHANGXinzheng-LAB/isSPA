#!/usr/bin/env python3
import os, sys, argparse

#from tqdm import tqdm

def replace_and_copy(src_path, dst_path, old_str, new_str, encoding='utf-8', buffer_size=65536):
    """
    复制文件并进行全局字符串替换
    
    参数：
    src_path  : 源文件路径
    dst_path  : 目标文件路径
    old_str   : 需要替换的字符串
    new_str   : 替换后的字符串
    encoding  : 文件编码 (默认 utf-8)
    buffer_size: 缓冲区大小 (字节数)
    """
    try:
        # 检查源文件是否存在
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"源文件 {src_path} 不存在")
            
        # 获取文件大小用于进度显示
        total_size = os.path.getsize(src_path)
        
        # 创建目标目录
        #os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # 使用二进制模式读写以保持精确控制
        with open(src_path, 'rb') as src_file, \
             open(dst_path, 'wb') as dst_file:
            
            # 初始化进度条
            #with tqdm(total=total_size, unit='B', unit_scale=True, desc="处理进度") as pbar:
                # 初始化缓冲区
            carry_over = b''
            
            while True:
                # 读取数据块
                chunk = src_file.read(buffer_size)
                if not chunk:
                    break
                
                # 合并上次剩余字节
                chunk = carry_over + chunk
                
                # 查找最后一个换行符位置
                last_newline = chunk.rfind(b'\n')
                
                if last_newline != -1:
                    # 分割可处理部分和剩余部分
                    process_part = chunk[:last_newline+1]
                    carry_over = chunk[last_newline+1:]
                else:
                    # 没有换行符则暂存全部
                    process_part = b''
                    carry_over = chunk
                
                # 转换为字符串并替换
                try:
                    decoded = process_part.decode(encoding)
                    replaced = decoded.replace(old_str, new_str)
                    encoded = replaced.encode(encoding)
                except UnicodeDecodeError:
                    raise ValueError(f"解码失败，请检查编码设置 ({encoding})")
                
                # 写入处理后的数据
                dst_file.write(encoded)
                #pbar.update(len(process_part))
            
            # 处理剩余字节
            if carry_over:
                try:
                    decoded = carry_over.decode(encoding)
                    replaced = decoded.replace(old_str, new_str)
                    encoded = replaced.encode(encoding)
                    dst_file.write(encoded)
                except UnicodeDecodeError:
                    raise ValueError("文件末尾存在不完整编码序列")
                #pbar.update(len(carry_over))
        
        print(f"文件处理完成: {src_path} -> {dst_path}")
        return True
        
    except Exception as e:
        print(f"\n处理失败：{str(e)}")
        # 清理可能生成的不完整文件
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False

if __name__ == "__main__":
    # 命令行参数配置
    parser = argparse.ArgumentParser(description="文件复制与全局替换工具")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("output", help="输出文件路径")
    parser.add_argument("--old", required=True, help="需要替换的字符串")
    parser.add_argument("--new", required=True, help="替换后的字符串")
    parser.add_argument("--encoding", default="utf-8", 
                       help="文件编码 (默认: utf-8)")
    parser.add_argument("--buffer", type=int, default=4096,
                       help="缓冲区大小 (字节，默认: 4096)")
    
    args = parser.parse_args()
    
    # 执行替换操作
    success = replace_and_copy(
        src_path=args.input,
        dst_path=args.output,
        old_str=args.old,
        new_str=args.new,
        encoding=args.encoding,
        buffer_size=args.buffer
    )
    
    # 退出状态码
    sys.exit(0 if success else 1)