#!/usr/bin/env python3

import os
import argparse
from collections import defaultdict

def calculate_molecular_weight_from_mmcif(file_path):
    """
    从mmCIF文件中读取原子类型并计算分子量总和
    
    参数:
    file_path (str): mmCIF文件的路径
    
    返回:
    float: 分子量总和
    """
    # 原子类型到分子量的映射（使用标准原子量）
    atomic_weights = {
        'H': 1.008, 'He': 4.0026, 'Li': 6.94, 'Be': 9.0122, 'B': 10.81,
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
        'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
        'S': 32.06, 'Cl': 35.45, 'Ar': 39.95, 'K': 39.098, 'Ca': 40.078,
        'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
        'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
        'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904,
        'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
        'Nb': 92.906, 'Mo': 95.95, 'Tc': 98.0, 'Ru': 101.07, 'Rh': 102.91,
        'Pd': 106.42, 'Ag': 107.87, 'Cd': 112.41, 'In': 114.82, 'Sn': 118.71,
        'Sb': 121.76, 'Te': 127.60, 'I': 126.90, 'Xe': 131.29, 'Cs': 132.91,
        'Ba': 137.33, 'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24,
        'Pm': 145.0, 'Sm': 150.36, 'Eu': 151.96, 'Gd': 157.25, 'Tb': 158.93,
        'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Yb': 173.05,
        'Lu': 174.97, 'Hf': 178.49, 'Ta': 180.95, 'W': 183.84, 'Re': 186.21,
        'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08, 'Au': 196.97, 'Hg': 200.59,
        'Tl': 204.38, 'Pb': 207.2, 'Bi': 208.98, 'Po': 209.0, 'At': 210.0,
        'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.04,
        'Pa': 231.04, 'U': 238.03, 'Np': 237.0, 'Pu': 244.0, 'Am': 243.0,
        'Cm': 247.0, 'Bk': 247.0, 'Cf': 251.0, 'Es': 252.0, 'Fm': 257.0,
        'Md': 258.0, 'No': 259.0, 'Lr': 266.0, 'Rf': 267.0, 'Db': 268.0,
        'Sg': 269.0, 'Bh': 270.0, 'Hs': 277.0, 'Mt': 278.0, 'Ds': 281.0,
        'Rg': 282.0, 'Cn': 285.0, 'Nh': 286.0, 'Fl': 289.0, 'Mc': 290.0,
        'Lv': 293.0, 'Ts': 294.0, 'Og': 294.0
    }
    
    total_weight = 0.0
    atom_count = defaultdict(int)
    index = 0
    
    try:
        with open(file_path, 'r') as file:
            in_atom_site_section = False
            
            for line in file:
                # 检查是否进入_atom_site部分
                if line.startswith('_atom_site.'):
                    in_atom_site_section = True
                    index += 1
                    # 查找type_symbol列的位置
                    if 'type_symbol' in line:
                        #type_symbol_index = line.split().index('_atom_site.type_symbol')
                        #print(line.split().index('_atom_site.type_symbol'))
                        type_symbol_index = index - 1
                    continue
                
                # 如果不在_atom_site部分，继续查找
                if not in_atom_site_section:
                    continue
                
                # 如果遇到新的部分，停止读取
                if line.startswith('_') and not line.startswith('_atom_site.'):
                    break
                
                # 跳过空行和注释行
                if not line.strip() or line.startswith('#'):
                    continue
                
                # 处理数据行
                data = line.split()
                if len(data) > type_symbol_index:
                    atom_type = data[type_symbol_index]
                    
                    # 处理可能的多字符原子类型（如"Fe"、"Zn"等）
                    # 只取第一个字符作为元素符号（如"Fe"变成"F"是错误的，需要处理）
                    # 这里我们检查原子类型是否在预定义的原子量字典中
                    if atom_type in atomic_weights:
                        weight = atomic_weights[atom_type]
                        total_weight += weight
                        atom_count[atom_type] += 1
                    else:
                        # 尝试转换为标准格式（首字母大写，其余小写）
                        standard_type = atom_type.capitalize()
                        if standard_type in atomic_weights:
                            weight = atomic_weights[standard_type]
                            total_weight += weight
                            atom_count[standard_type] += 1
                        else:
                            print(f"警告: 未知原子类型 '{atom_type}'，已跳过")
                
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return None
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None
    
    # 打印原子统计信息
    print("原子统计:")
    total_atoms = 0
    for atom, count in sorted(atom_count.items()):
        total_atoms += count
        print(f"  {atom}: {count} 个原子")
    print(f"  总共: {total_atoms} 个原子")
    
    return total_weight

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算模型的分子量')
    parser.add_argument('input', help='原子模型的.CIF文件')

    args = parser.parse_args()

    mmcif_file = args.input
    
    if os.path.exists(mmcif_file):
        molecular_weight = calculate_molecular_weight_from_mmcif(mmcif_file)
        if molecular_weight is not None:
            print(f"\n总分子量: {molecular_weight:.2f}")
    else:
        print(f"文件 '{mmcif_file}' 不存在，请提供有效的mmCIF文件路径")