#!/usr/bin/env python3

import os
import mrcfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # 用于进度条显示
from collections import defaultdict

# 之前定义的函数保持不变 (count_particles_per_micrograph, save_counts_to_csv, plot_particle_distribution, filter_micrographs_by_count, generate_summary_report)

def count_particles_per_micrograph(star_file_path):
    """
    统计STAR文件中每张照片的颗粒数量
    
    参数:
        star_file_path: STAR文件路径
        
    返回:
        包含每张照片颗粒数量的字典 {微镜图像文件名: 颗粒数量}
    """
    # 初始化计数器
    micrograph_counts = defaultdict(int)
    in_particles_section = False
    columns = []
    micrograph_col_index = -1
    
    with open(star_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
                
            # 检测数据块开始
            if line.startswith('data_'):
                section_name = line.split('_', 1)[1]
                in_particles_section = (section_name == 'particles')
                columns = []  # 重置列定义
                continue
                
            # 检测循环开始
            if line == 'loop_' and in_particles_section:
                columns = []
                continue
                
            # 解析字段定义
            if line.startswith('_rln') and in_particles_section and columns is not None:
                # 提取字段名和列索引
                parts = line.split()
                field_name = parts[0]
                columns.append(field_name)
                
                # 记录微镜图像名称的列索引
                if field_name == '_rlnMicrographName':
                    micrograph_col_index = len(columns) - 1
                continue
                
            # 解析数据行
            if in_particles_section and columns and micrograph_col_index != -1:
                data = line.split()
                if len(data) > micrograph_col_index:
                    micrograph_name = data[micrograph_col_index]
                    micrograph_counts[micrograph_name] += 1
    
    return dict(micrograph_counts)

def save_counts_to_csv(counts_dict, output_path):
    """
    将颗粒数量统计保存到CSV文件
    
    参数:
        counts_dict: 包含颗粒数量的字典
        output_path: 输出文件路径
    """
    # 转换为DataFrame以便保存
    df = pd.DataFrame({
        'Micrograph': list(counts_dict.keys()),
        'ParticleCount': list(counts_dict.values())
    })
    
    # 按颗粒数量排序
    df = df.sort_values(by='ParticleCount', ascending=False)
    
    # 保存为CSV
    df.to_csv(output_path, index=False)
    print(f"已保存统计结果到: {output_path}")
    
    return df

def plot_particle_distribution(df, output_dir):
    """
    生成颗粒数量分布直方图
    
    参数:
        df: 包含颗粒数量统计的DataFrame
        output_dir: 输出目录
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置输出路径
    plot_path = os.path.join(output_dir, "particle_distribution.png")
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    
    # 计算合适的bins数量
    max_count = df['ParticleCount'].max()
    min_count = df['ParticleCount'].min()
    bin_width = max(1, int((max_count - min_count) / 20))
    bins = np.arange(min_count, max_count + bin_width, bin_width)
    
    # 绘制直方图
    plt.hist(df['ParticleCount'], bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 添加统计信息
    mean_count = df['ParticleCount'].mean()
    median_count = df['ParticleCount'].median()
    plt.axvline(mean_count, color='red', linestyle='dashed', linewidth=1, label=f'mean: {mean_count:.1f}')
    plt.axvline(median_count, color='green', linestyle='dashed', linewidth=1, label=f'median: {median_count:.1f}')
    
    # 设置图表属性
    plt.title('Histogram of Particle Number', fontsize=14)
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel('Number of Micrographs', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"已生成颗粒分布直方图: {plot_path}")
    return plot_path

def filter_micrographs_by_count(df, min_count, max_count, output_dir):
    """
    筛选特定颗粒数量范围的照片
    
    参数:
        df: 包含颗粒数量统计的DataFrame
        min_count: 最小颗粒数量
        max_count: 最大颗粒数量
        output_dir: 输出目录
        
    返回:
        筛选后的DataFrame
    """
    # 应用筛选条件
    filtered_df = df[(df['ParticleCount'] >= min_count) & (df['ParticleCount'] <= max_count)]
    
    # 保存筛选结果
    output_path = os.path.join(output_dir, f"filtered_{min_count}-{max_count}.csv")
    filtered_df.to_csv(output_path, index=False)
    
    print(f"\n筛选结果 (颗粒数量范围: {min_count}-{max_count}):")
    print(f"符合条件照片数量: {len(filtered_df)}")
    print(f"照片中颗粒总数: {filtered_df['ParticleCount'].sum()}")
    print(f"已保存筛选结果到: {output_path}")
    
    return filtered_df

def generate_summary_report(df, filtered_df, output_dir):
    """
    生成分析报告
    
    参数:
        df: 原始DataFrame
        filtered_df: 筛选后的DataFrame
        output_dir: 输出目录
    """
    report_path = os.path.join(output_dir, "analysis_report.txt")
    
    total_micrographs = len(df)
    total_particles = df['ParticleCount'].sum()
    min_count = df['ParticleCount'].min()
    max_count = df['ParticleCount'].max()
    avg_count = total_particles / total_micrographs if total_micrographs > 0 else 0
    
    with open(report_path, 'w') as f:
        f.write("===== 颗粒数量统计分析报告 =====\n\n")
        f.write(f"微镜图像总数: {total_micrographs}\n")
        f.write(f"颗粒总数: {total_particles}\n")
        f.write(f"每张照片平均颗粒数: {avg_count:.1f}\n")
        f.write(f"单张照片最小颗粒数: {min_count}\n")
        f.write(f"单张照片最大颗粒数: {max_count}\n")
        f.write(f"颗粒数量中位数: {df['ParticleCount'].median():.1f}\n")
        f.write(f"颗粒数量标准差: {df['ParticleCount'].std():.1f}\n\n")
        
        if filtered_df is not None:
            f.write("===== 筛选结果 =====\n")
            f.write(f"筛选条件: 颗粒数量范围 {filtered_df['ParticleCount'].min()}-{filtered_df['ParticleCount'].max()}\n")
            f.write(f"符合条件的照片数量: {len(filtered_df)}\n")
            f.write(f"照片中颗粒总数: {filtered_df['ParticleCount'].sum()}\n")
            f.write(f"占总照片比例: {len(filtered_df)/total_micrographs*100:.1f}%\n")
            f.write(f"占总颗粒比例: {filtered_df['ParticleCount'].sum()/total_particles*100:.1f}%\n")
    
    print(f"已生成分析报告: {report_path}")

def calculate_average_intensity(mrc_path):
    """
    计算MRC文件的像素强度平均值
    
    参数:
        mrc_path: MRC文件路径
        
    返回:
        像素强度平均值
    """
    try:
        with mrcfile.open(mrc_path, mode='r') as mrc:
            data = mrc.data
            
            # 处理不同维度的数据
            if len(data.shape) == 3:  # 3D数据，取第一层
                data = data[0]
            elif len(data.shape) > 3:  # 处理异常维度
                raise ValueError(f"不支持的维度: {data.shape}")
                
            # 计算平均值
            return np.mean(data)
    except Exception as e:
        print(f"处理文件 {mrc_path} 时出错: {str(e)}")
        return np.nan

def calculate_intensities_for_micrographs(micrograph_paths, base_dir):
    """
    为多个微镜图像计算像素强度平均值
    
    参数:
        micrograph_paths: 微镜图像路径列表
        base_dir: 基础目录，用于构建完整路径
        
    返回:
        包含路径和平均强度的字典 {微镜图像路径: 平均强度}
    """
    intensities = {}
    
    print(f"\n开始计算 {len(micrograph_paths)} 张照片的像素强度平均值...")
    
    # 使用进度条
    for path in tqdm(micrograph_paths, desc="处理照片"):
        # 构建完整路径
        full_path = os.path.join(base_dir, path) if base_dir else path
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            print(f"警告: 文件不存在 - {full_path}")
            intensities[path] = np.nan
            continue
            
        # 计算平均强度
        avg_intensity = calculate_average_intensity(full_path)
        intensities[path] = avg_intensity
    
    return intensities

def plot_intensity_distribution(intensities, output_dir):
    """
    绘制像素强度平均值分布直方图
    
    参数:
        intensities: 包含平均强度的字典
        output_dir: 输出目录
    """
    # 提取有效强度值 (排除NaN)
    valid_intensities = [v for v in intensities.values() if not np.isnan(v)]
    
    if not valid_intensities:
        print("警告: 没有有效的强度值可用于绘图")
        return None
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置输出路径
    plot_path = os.path.join(output_dir, "intensity_distribution.png")
    
    # 创建图表
    plt.figure(figsize=(12, 7))
    
    # 计算合适的bins数量
    min_val = min(valid_intensities)
    max_val = max(valid_intensities)
    bin_width = (max_val - min_val) / 30  # 使用30个bins
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    
    # 绘制直方图
    plt.hist(valid_intensities, bins=bins, color='lightcoral', edgecolor='black', alpha=0.8)
    
    # 添加统计信息
    mean_val = np.mean(valid_intensities)
    median_val = np.median(valid_intensities)
    plt.axvline(mean_val, color='blue', linestyle='dashed', linewidth=2, label=f'mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'median: {median_val:.2f}')
    
    # 设置图表属性
    plt.title('Histogram of average intensity', fontsize=16)
    plt.xlabel('Average Intensity', fontsize=14)
    plt.ylabel('Number of Micrographs', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加统计摘要文本框
    stats_text = f"""
    统计摘要:
    照片数量: {len(valid_intensities)}
    最小值: {min_val:.2f}
    最大值: {max_val:.2f}
    平均值: {mean_val:.2f}
    标准差: {np.std(valid_intensities):.2f}
    中位数: {median_val:.2f}
    """
    plt.annotate(stats_text, xy=(0.98, 0.7), xycoords='axes fraction', 
                fontsize=12, ha='right', va='top', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"已生成强度分布直方图: {plot_path}")
    return plot_path

def save_intensity_results(df, output_dir):
    """
    保存强度计算结果
    
    参数:
        df: 包含结果的DataFrame
        output_dir: 输出目录
    """
    output_path = os.path.join(output_dir, "micrograph_intensities.csv")
    df.to_csv(output_path, index=False)
    print(f"已保存强度计算结果到: {output_path}")
    return output_path

# 主程序
if __name__ == "__main__":
    # 配置参数
    #star_file_path = "your_star_file.star"  # 替换为你的STAR文件路径
    star_file_path = input("\n请输入颗粒的STAR文件: ")
    output_dir = input("\n请输入输出路径: ")
    #output_dir = "particle_analysis"        # 输出目录
    micrographs_base_dir = input("\n请输入照片所在的路径: ")  # 微镜图像基础目录
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 统计颗粒数量
    print(f"正在统计STAR文件中的颗粒数量: {star_file_path}")
    counts = count_particles_per_micrograph(star_file_path)
    
    if not counts:
        print("未找到颗粒数据或无法解析STAR文件")
        exit()
    
    print(f"成功统计了 {len(counts)} 张照片中的颗粒数量")
    
    # 保存统计结果
    counts_csv = os.path.join(output_dir, "particle_counts.csv")
    counts_df = save_counts_to_csv(counts, counts_csv)
    
    # 生成颗粒数量分布直方图
    plot_particle_distribution(counts_df, output_dir)
    
    # 筛选特定颗粒数量范围的照片
    min_count = int(input("\n请输入筛选的最小颗粒数量: ") or 0)
    max_count = int(input("请输入筛选的最大颗粒数量: ") or counts_df['ParticleCount'].max())
    
    filtered_df = filter_micrographs_by_count(counts_df, min_count, max_count, output_dir)
    
    # 生成分析报告
    generate_summary_report(counts_df, filtered_df, output_dir)
    
    # 获取筛选后的照片路径列表
    micrograph_paths = filtered_df['Micrograph'].tolist()
    
    # 计算像素强度平均值
    intensities = calculate_intensities_for_micrographs(micrograph_paths, micrographs_base_dir)
    
    # 将结果添加到DataFrame
    filtered_df['AverageIntensity'] = filtered_df['Micrograph'].map(intensities)
    
    # 保存结果
    save_intensity_results(filtered_df, output_dir)
    
    # 绘制强度分布直方图
    plot_intensity_distribution(intensities, output_dir)
    
    # 打印摘要信息
    valid_intensities = [v for v in intensities.values() if not np.isnan(v)]
    if valid_intensities:
        print("\n===== 像素强度摘要 =====")
        print(f"照片数量: {len(valid_intensities)}")
        print(f"平均强度范围: {min(valid_intensities):.2f} - {max(valid_intensities):.2f}")
        print(f"全局平均值: {np.mean(valid_intensities):.2f}")
        print(f"全局标准差: {np.std(valid_intensities):.2f}")
    
    print("\n===== 分析完成 =====")
    print(f"所有结果已保存到目录: {output_dir}")