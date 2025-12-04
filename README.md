# isSPA_1.2

In the version, GUI is added and several bugs are fixed.

## Update

1. Now isSPA can be run on different GPUs (5090, 4090, 3080, 3060, 2080 Ti and A100 have been tested).

## Installation
1.	Install virtual environment module venv (the python version should be adjusted according to your system):
```
sudo apt install python3.10-venv
```
2.	Create virtual environment in your preferred directory:
```
cd /home/user/Software/
python3 -m venv isSPA_env
```
3.	Activate the virtual environment:
```
source /home/user/Software/isSPA_env/bin/activate
```
4.	Install required libraries:
```
pip install PyQt6 nvidia-ml-py mrcfile scipy
```
5.  Download HDF5 package from the official website: https://support.hdfgroup.org/downloads/index.html 
6.	Uncompress it
```
tar -xzf hdf5-1.14.6.tar.gz
```
7.	Install HDF5 according to **./hdf5-1.14.6/release_docs/INSTALL_Autotools.txt**.
8.	(Recommended) Add the absolute path of the library of HDF5 to *LD_LIBRARY_PATH*. For instance,
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/Software/hdf5/lib
```
9.	Enter the directory of isSPA_1.2, and modify *LIB_HDF5* and *INCLUDE_HDF5* in the **Makefile** according to the installation paths in step 7.
10.	Edit the first line of **Makefile**, making sure the path of *SHELL* is correct in your system and *-arch=sm_86* corresponds to your GPUs. GPU Compute Capability can be found here: https://developer.nvidia.com/cuda-gpus
11.	Execute the following commands (N is the number of available threads):
```
make -j N
make install
```
12.  (Recommended) Add the absolute paths of **./isSPA_1.2/build** and **./isSPA_1.2/isSPA_scripts** to environment variables. For example,
```
export PATH=/home/user/Software/isSPA_1.2/build:$PATH
export PATH=/home/user/Software/isSPA_1.2/isSPA_scripts:$PATH
```

## Answers to some frequently asked questions
1. Which software should I use to generate projection files?
It automatically uses RELION now. You can use EMAN1 or EMAN2 if you prefer.
2. What version of Python should I use?
**Python 3**.
3. How does isSPA use multiple GPUs?
You can use multiplt GPUs just by selecting them in the GUI now.


## Contributor
Li Rui, Chen Yuanbo, Zhao Mingjie, Cheng Yuanhao

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FZHANGXinzheng-LAB%2FisSPA_1.1.2&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=VIEWS&edge_flat=false)](https://hits.seeyoufarm.com)
