#!/usr/bin/env python3
# coding: utf-8

import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    ap = argparse.ArgumentParser(description="interatively explore score threshold to split the Gaussian mixtures")
    ap.add_argument("-i", "--input", metavar = 'PATH', required = True, help = ".npy file containing particle scores")
    ap.add_argument("-n", metavar = 'N', type=int, default=100, help = "number of bins (default to %(default)s)")

    args = ap.parse_args()
    data = np.load(args.input)
    plt.figure(figsize = (10, 6.18))
    sns.histplot(data, bins = args.n, kde = True)
    plt.xlabel('phase residual')
    plt.ion()
    plt.pause(99999)

if __name__ == "__main__":
    main()
