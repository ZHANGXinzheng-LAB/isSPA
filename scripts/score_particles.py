#!/usr/bin/env python3
# coding: utf-8

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mrcfile
import starfile

def make_real_space_mask(height, width, diameter, edge): 
    h0, w0 = height, width
    if diameter is None: diameter = h0
    my, mx = np.indices((h0, w0), dtype=np.float32)
    cy, cx = h0 / 2.0, w0 / 2.0
    r_dist = np.hypot(mx - cx, my - cy)
    
    rmask = np.ones((h0, w0), dtype=np.float32)
    r_out = diameter / 2.0
    r_in = max(0.0, r_out - edge)
    
    if edge > 0:
        edge_area = (r_dist > r_in) & (r_dist <= r_out)
        rmask[edge_area] = (
            0.5 * np.cos(math.pi * (r_dist[edge_area] - r_in) / edge) + 0.5
        ).astype(np.float32)
        
    rmask[r_dist > r_out] = 0.0
    return rmask

def build_frc_radius_bins(l: int) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute radius-bin mapping for a fixed l x l grid."""
    fx, fy = np.fft.fftfreq(l)*l, np.fft.fftfreq(l)*l
    radius = np.hypot(*np.meshgrid(fy, fx, indexing="ij"))
    shell = np.fft.fftshift( np.rint(radius).astype(int) )
    valid = (shell >= 0) & (shell < l//2)
    return shell[valid].ravel(), valid.ravel()

def score_one(img, tmp, rid, valid_flat) -> np.ndarray:
    """Compute FRC curve between two 2D projections.

    内联版本（不依赖外部 cal_frc.py）：
    - 保持原始环半径定义；
    - 仅累计 r 本身（不累计 r±1..±3）；
    - 采用预计算半径分桶 + 向量化 bincount。
    """
    fimg = np.fft.fftshift(np.fft.fft2(img))
    ftmp = np.fft.fftshift(np.fft.fft2(tmp))
    aimg = np.angle(fimg.ravel()[valid_flat])
    atmp = np.angle(ftmp.ravel()[valid_flat])
    phase_residual = np.cos(aimg-atmp)
    return np.bincount(rid, phase_residual) / np.bincount(rid)

def score_many(df, projections, rmask, ncpu, dir): 
    N = len(df)
    frc_rid, frc_valid_flat = build_frc_radius_bins(projections.shape[1])
    pool = ThreadPoolExecutor(ncpu)
    path, mrc, fuL = '', None, []
    for i in range(N): 
        i2, path2 = df['rlnImageName'][i].split('@')
        if path != path2: path, mrc = path2, mrcfile.open(dir+'/'+path2)
        img, tmp = mrc.data[int(i2)-1]*rmask, projections[i]*rmask
        fuL.append(pool.submit(score_one, img, tmp, frc_rid, frc_valid_flat))
    score = np.stack([fu.result() for fu in fuL])
    return score

def average_Fring(score: np.ndarray, apix, lowres, highres) -> np.ndarray:
    R = score.shape[-1]
    kvals = np.arange(0, R, dtype=np.float64) / (2*R) / float(apix)
    mask = (kvals >= 1/lowres) & (kvals <= 1/highres)
    if sum(mask)==0: return np.zeros(len(score))
    return score[:, mask].mean(axis = 1)

def plot(path): 
    data = np.load(path)
    plt.figure(figsize = (10, 6.18))
    sns.histplot(data, bins = 100, kde = True)
    plt.xlabel('phase residual')
    plt.savefig(path.replace('.npy', '.png'), dpi = 300, bbox_inches = 'tight')
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Rescore particles by Fourier phase residual")
    ap.add_argument("-i", "--input", metavar = 'PATH', required = True, help = "particles STAR file containing blob path, CTF, orientation")
    ap.add_argument("-r", "--projection", metavar = 'PATH', required = True, help = "projections of reconstructed 3D volume")
    ap.add_argument('-d', "--dir", metavar = 'PATH', default = '.', help = "directory to add before rlnImageName (default to %(default)s)")    
    ap.add_argument("-o", "--out", metavar = 'PATH', help = "Output .npy path")
    ap.add_argument("-j", '--cpu', metavar = 'N', type=int, help="Number of CPUs to use")
    ap.add_argument("--highres", metavar = 'R', type=float, default = 6.0, help = "highest resolution for computing (default to %(default)s Å)")
    ap.add_argument("--lowres", metavar = 'R', type=float, default = 400.0, help = "lowest resolution for computing (default to %(default)s Å)")
    ap.add_argument("--mask-diameter",metavar = 'R', type=float, default = None, help="Real space circular mask diameter in pixels")
    ap.add_argument("--mask-edge", metavar = 'R', type=float, default = 6.0, help="Real space circular mask soft edge width (default to %(default)s px)")

    args = ap.parse_args()
    if args.out is None: args.out = args.input.replace('.star', '.npy')
    star = starfile.read(args.input) 
    apix = star['optics']['rlnImagePixelSize'][0]
    projections = mrcfile.open(args.projection).data
    assert star['optics']['rlnImageSize'][0] == projections.shape[1], 'particle and template must have the same shape for FRC'
    assert projections.shape[1] == projections.shape[2], f"Only square images are supported now: got {projections.shape[1:]}"

    rmask = make_real_space_mask(*projections.shape[1:], args.mask_diameter, args.mask_edge)
    score = score_many(star['particles'], projections, rmask, args.cpu, args.dir)
    mscore = average_Fring(score, apix, args.lowres, args.highres)
    np.save(args.out, mscore), plot(args.out)

if __name__ == "__main__":
    main()
