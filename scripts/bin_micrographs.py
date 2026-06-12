#!/usr/bin/env python3

import argparse
import os
import subprocess

"""
缩小所有照片的尺寸并放到一个文件夹中
"""
parser = argparse.ArgumentParser(description='This program will bin all the micrographs and put them in a directory.')
parser.add_argument('input', metavar='Micrograph_directory', help='The directory containing micrographs')
parser.add_argument('bin', metavar='Bin_factor', type=int, help='The bin factor you would like to use')
parser.add_argument('pixel', metavar='Pixel_size', type=float, help='The original pixel size')
parser.add_argument('-s', metavar='--suffix', dest='suffix', default='.mrc', help='The suffix of micrographs')

args = parser.parse_args()

input_dir = args.input
bin_factor = args.bin
pixel = args.pixel
suffix = args.suffix

pixel_b = pixel * bin_factor

file_list = []
for file in os.listdir(input_dir):
    if file.endswith(suffix):
        file_list.append(file)

# 创建文件夹
if os.path.exists(f'{input_dir}/bin{bin_factor}') == False:
    os.mkdir(f'{input_dir}/bin{bin_factor}')

for i in file_list:
    subprocess.Popen(f'relion_image_handler --i {input_dir}/{i} --o {input_dir}/bin{bin_factor}/{i} --angpix {pixel} --rescale_angpix {pixel_b}', shell=True)