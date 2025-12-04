#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import auc

from cal_euler_distance import cal_euler_distance
from parse_star import parse_star

def convert_numeric(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            # 如果转换失败，保持原样
            continue
    return df

def calculate_pr(truth_df, pred_df, distance_tol=4, angle_tol=6, confidence_col='AutopickFigureOfMerit', output_matches=None):
    """主计算函数"""
    # 数据预处理
    truth_df = convert_numeric(truth_df)
    pred_df = convert_numeric(pred_df)
    pred_df = pred_df.sort_values(confidence_col, ascending=False)

    # 初始化全局统计
    total_true = len(truth_df)
    total_pred = len(pred_df)

    # 按微图分组处理
    truth_groups = truth_df.groupby('MicrographName')
    pred_groups = pred_df.groupby('MicrographName')
    
    # 存储匹配状态和置信度序列
    matches = []
    confidence_scores = []

    # 第一遍：收集所有可能的匹配
    for mg_name, pred_group in tqdm(pred_groups, desc="预匹配"):
        if mg_name not in truth_groups.groups:
            continue
            
        truth_group = truth_groups.get_group(mg_name)
        truth_coords = truth_group[['CoordinateX', 'CoordinateY']].values.astype(float)
        truth_angles = truth_group[['AngleRot', 'AngleTilt', 'AnglePsi']].values.astype(float)
        
        # 构建索引结构
        kd_tree = KDTree(truth_coords)
        
        for _, pred_row in pred_group.iterrows():
            x = float(pred_row['CoordinateX'])
            y = float(pred_row['CoordinateY'])
            confidence = float(pred_row[confidence_col])
            pred_angle = pred_row[['AngleRot', 'AngleTilt', 'AnglePsi']].values.astype(float)
            
            # 坐标匹配
            dists, indices = kd_tree.query([[x, y]], k=10)  # 检查前10个最近邻
            for dist, idx in zip(dists[0], indices[0]):
                if dist > distance_tol:
                    continue
                
                # 角度匹配
                truth_angle = truth_angles[idx]
                angle_diff = cal_euler_distance(truth_angle, pred_angle)
                if angle_diff <= angle_tol:
                    matches.append({
                        'truth_idx': truth_group.index[idx],
                        'pred_idx': pred_row.name,
                        'confidence': confidence
                    })
                    break

    # 按置信度降序排序匹配项
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    #print(matches)
    
    # 第二遍：确定唯一匹配
    truth_matched = set()
    pred_matched = set()
    true_positives = []
    match_records = []  # 存储匹配记录
    
    for match in matches:
        if match['truth_idx'] not in truth_matched and match['pred_idx'] not in pred_matched:
            truth_row = truth_df.loc[match['truth_idx']]
            pred_row = pred_df.loc[match['pred_idx']]
            
            record = {
                'micrograph': pred_row['MicrographName'],
                'truth_x': float(truth_row['CoordinateX']),
                'truth_y': float(truth_row['CoordinateY']),
                'pred_x': float(pred_row['CoordinateX']),
                'pred_y': float(pred_row['CoordinateY']),
                'confidence': match['confidence'],
                'truth_rot': float(truth_row['AngleRot']),
                'truth_tilt': float(truth_row['AngleTilt']),
                'truth_psi': float(truth_row['AnglePsi']),
                'pre_rot': float(pred_row['AngleRot']),
                'pre_tilt': float(pred_row['AngleTilt']),
                'pre_psi': float(pred_row['AnglePsi']),
                'truth_dfu': float(truth_row['DefocusU']),
                'pre_dfu': float(pred_row['DefocusU'])
            }
            match_records.append(record)
            true_positives.append(match['confidence'])
            truth_matched.add(match['truth_idx'])
            pred_matched.add(match['pred_idx'])

    if output_matches:
        write_match_records(match_records, output_matches)

    # 统计最终结果
    tp = len(true_positives)
    fp = total_pred - tp
    fn = total_true - tp
    
    # 生成PR曲线数据点
    confidences = np.array([m['confidence'] for m in match_records] + [0])
    precisions = []
    recalls = []

    for thresh in np.unique(confidences):
        above = [c >= thresh for c in true_positives]
        current_tp = sum(above)
        current_fp = sum(pred_df[confidence_col] >= thresh) - current_tp
        
        precision = current_tp / (current_tp + current_fp) if (current_tp + current_fp) > 0 else 1.0

        recall = current_tp / total_true
        
        precisions.append(precision)
        recalls.append(recall)
    
    # 添加基准点
    #precisions.append(1.0)
    #recalls.append(0.0)
    
    # 按Recall排序
    sort_idx = np.argsort(recalls)
    recalls = np.array(recalls)[sort_idx]
    precisions = np.array(precisions)[sort_idx]
    
    # 计算AUC
    auprc = auc(recalls, precisions)
    
    return precisions, recalls, auprc, {'TP': tp, 'FP': fp, 'FN': fn}

def plot_pr_curve(precision, recall, auprc, stats, output_file):
    """增强的可视化函数"""
    plt.figure(figsize=(7,5))
    
    # PR曲线
    plt.plot(recall, precision, label=f'PR Curve (AUPRC = {auprc:.3f})', lw=2)
    
    # 关键点标注
    max_f1 = 0
    best_thresh = 0
    for p, r in zip(precision, recall):
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > max_f1:
            max_f1 = f1
            best_thresh = p
    
    #plt.scatter([recall[np.argmax(precision)]], [np.max(precision)], color='red', zorder=10, label=f'Max Precision: {np.max(precision):.2f}')
    
    # 统计信息框
    textstr = '\n'.join([
        f'Total True: {stats["TP"] + stats["FN"]}',
        f'Total Pred: {stats["TP"] + stats["FP"]}',
        f'TP: {stats["TP"]}',
        f'FP: {stats["FP"]}',
        f'FN: {stats["FN"]}',
        f'F1 Score: {max_f1:.3f}'
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.4)
    plt.gca().text(0.7, 0.4, textstr, transform=plt.gca().transAxes,
                  verticalalignment='top', bbox=props)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.title('isSPA Picking Performance', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.2)
    plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    plt.tick_params(which='both', direction='in')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def write_match_records(records, output_path):
    """将匹配记录写入CSV文件"""
    import csv
    
    fieldnames = [
        'micrograph', 
        'truth_x', 'truth_y',
        'pred_x', 'pred_y',
        'confidence', 
        'truth_rot', 'truth_tilt', 'truth_psi', 
        'pre_rot', 'pre_tilt', 'pre_psi',
        'truth_dfu', 'pre_dfu'

    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for record in records:
            # 格式化浮点数
            formatted = {k: f"{v:.2f}" if isinstance(v, float) else v for k, v in record.items()}
            writer.writerow(formatted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算isSPA的PR曲线')
    parser.add_argument('truth_star', help='真实颗粒的STAR文件')
    parser.add_argument('pred_star', help='挑选颗粒的STAR文件')
    parser.add_argument('--output', default='pr_curve.png', help='输出图像路径')
    parser.add_argument('--text', default='pr_data.txt', help='输出数据路径')
    parser.add_argument('--distance_tol', type=float, default=4, help='坐标匹配容差 (像素)')
    parser.add_argument('--angle_tol', type=float, default=6, help='角度匹配容差 (度)')
    parser.add_argument('--confidence_col', default='AutopickFigureOfMerit', help='置信度分数列名')
    parser.add_argument('--output_matches', default=None, help='匹配结果输出路径（CSV格式）')

    args = parser.parse_args()

    # 读取数据
    truth_df = parse_star(args.truth_star)
    pred_df = parse_star(args.pred_star)

    # 执行计算
    precision, recall, auprc, stats = calculate_pr(
        truth_df, pred_df,
        distance_tol=args.distance_tol,
        angle_tol=args.angle_tol, 
        confidence_col=args.confidence_col, 
        output_matches=args.output_matches
    )
    np.savetxt(args.text, np.array([precision, recall]), fmt='%.4f')
    plot_pr_curve(precision, recall, auprc, stats, args.output)
    
    print(f"图片已保存至 {args.output}")
    print(f"数据已保存至 {args.text}")
    print(f"关键统计：")
    print(f"- 真实粒子总数: {stats['TP'] + stats['FN']}")
    print(f"- 预测粒子总数: {stats['TP'] + stats['FP']}")
    print(f"- 正确匹配 (TP): {stats['TP']}")
    print(f"- 假阳性 (FP): {stats['FP']}")
    print(f"- 假阴性 (FN): {stats['FN']}")
    print(f"- AUPRC: {auprc:.4f}")