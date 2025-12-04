import argparse
import mrcfile

from parse_star import parse_star

def merge_maps(pdb, mrc, fsc):
    with mrcfile.open(pdb) as f:
        data1 = f.data
        head1 = f.header
        voxel_size1 = f.voxel_size
    with mrcfile.open(mrc) as f:
        data2 = f.data
        head2 = f.header
        voxel_size2 = f.voxel_size
    # 检查输入
    if voxel_size1 != voxel_size2:
        print("两张图的像素尺寸不一致！")
        return -1
    elif head1['nx'] != head2['nx'] or head1['ny'] != head2['ny'] or head1['nz'] != head2['nz']:
        print("两张图的尺寸不一致！")
        return -1
    fsc_df = parse_star(fsc, block='data_fsc')
    fsc_df = fsc_df.apply(pd.to_numeric, errors='ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合成模板')
    parser.add_argument('pdb', help='根据原子模型生成的电势图')
    parser.add_argument('mrc', help='中等分辨率的电势图')
    parser.add_argument('fsc', help='FSC文件')

    args = parser.parse_args()
    
    merge_maps(args.pdb, args.mrc, args.fsc)