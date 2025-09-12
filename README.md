# isSPA_1.1.2

In this version, several scripts are added to do the preprocess and postprocess.

## Update

1. We previously observed that isSPA might produce inconsistent results on different machines with varying GPU cards and CUDA versions. This issue has now been partially resolved.
  
  Experimental tests show that this version identifies more correct particles and achieves higher resolution under the same data processing methods (including Class3D, Refine3D, Postprocess, and other subsequent processing).

2. In this version, when **norm_type=1** is in the config, the computation is performed on the entire image. Experimental tests indicate that the results with **norm_type=1** are better than those with **norm_type=0**, with more particles identified.
When **norm_type=0** is used, the large image is divided into smaller (e.g.720x720) sections for computation, and the normalization algorithm is not applied. This setting is suitable for machines with less powerful GPUs, as processing large images can be demanding on the GPU, and omitting normalization can speed up the computation.

## Installation
1.	Download HDF5 package from the official website:
  https://support.hdfgroup.org/downloads/index.html 
2.	Uncompress it, such as
```
tar -xzf hdf5-1.14.5.tar.gz
```
4.	Install HDF5 according to **./hdf5-1.14.5/release_docs/INSTALL_Autotools.txt**.
5.	(Recommended) Add the absolute path of the library of HDF5 to *LD_LIBRARY_PATH*. For instance,
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/Software/hdf5/lib
```
6.	Enter the directory of isSPA_1.1.2, and modify *LIB_HDF5* and *INCLUDE_HDF5* in the **Makefile** according to the installation paths in step 3.
7.	Edit the first line of **Makefile**, making sure the path of *SHELL* is correct in your system.
8.	Execute the following commands (N is the number of available threads):
```
make -j N
make install
```
9.  (Recommended) Add the absolute paths of **./isSPA_1.1.2/build** and **./isSPA_1.1.2/isSPA_scripts** to environment variables. For example,
```
export PATH=/home/user/Software/isSPA_1.1.2/build:$PATH
export PATH=/home/user/Software/isSPA_1.1.2/isSPA_scripts:$PATH
```

## Answers to some frequently asked questions
1. Which software should I use to generate projection files?
You can use EMAN1, EMAN2 or RELION now.
2. What version of Python should I use?
**Python 3**.
3. How does isSPA use multiple GPUs?
You can split the entire dataset into multiple parts using the **first** and **last** parameters in the config file. Each part can then be processed on a separate GPU.

## Contributor
Li Rui, Chen Yuanbo, Zhao Mingjie, Cheng Yuanhao

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FZHANGXinzheng-LAB%2FisSPA_1.1.2&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=VIEWS&edge_flat=false)](https://hits.seeyoufarm.com)
