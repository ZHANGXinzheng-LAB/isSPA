# Installation

1. Download HDF5 package from the official website:  
  https://support.hdfgroup.org/downloads/index.html 
2. Uncompress it, such as `tar -xzf hdf5-1.14.5.tar.gz`
3. Install HDF5 according to `./hdf5-1.14.5/release_docs/INSTALL_Autotools.txt`
4. Enter `isSPA_1.1.2/`, and modify `LIB_HDF5` and `INCLUDE_HDF5` in the `Makefile` according to the installation paths in step 3.
5. Execute the following commands (N is the number of available threads):
   ```bash
   make -j N
   make install
   ```
6. (Recommended) Add the absolute paths of `./isSPA_1.1.2/build` and `./isSPA_1.1.2/scripts` to environment variables. For example,
   ```bash
   export PATH=/home/user/Software/isSPA_1.1.2/build:$PATH
   export PATH=/home/user/Software/isSPA_1.1.2/scripts:$PATH
   ```
7. (Recommended) Add the absolute path of the library of HDF5 to LD_LIBRARY_PATH. For instance,
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/Software/hdf5/lib
   ```

# Manual  

1. Use RELION to do Import and CTF estimation, generating `micrographs_ctf.star` file. Here we have 20 micrographs as a test dataset. The pixel size is 1.36. Voltage is 300 kV. Amplitude contrast is 0.07. 
2. Use the following command to bin all the micrographs and store them in another directory, converting STAR file to LST file which can be read by isSPA. Taking bin2 for example,   
   `preprocess.py micrographs/ 2 1.36 CtfFind/job002/micrographs_ctf.star`   
   The binned micrographs are stored in `./micrographs/bin2/`
3. If needed, adjust the pixel size of the 3D template:   
   `relion_image_handler --i 3D_template.mrc --o 3D_template_rescaled.mrc --angpix 1.36 --rescale_angpix 2.72`
4. Use EMAN2 to generate 2D projections of the 3D template:   
   `e2project3d.py  3D_template.mrc  --outfile=projections_5.hdf  --orientgen=eman:delta=5:inc_mirror=1 --sym=c1 --compressbits=0 --verbose=2 > eman2_c1_delta5_mirror.txt`
5. Convert the text file output by EMAN2 to LST file:   
   `euler_angles_txt_to_lst.py eman2_c1_delta5_mirror.txt`   
   The output file is `eman2_c1_delta5_mirror.lst`
6. Modify the `config` file and execute `isSPA config` 
   - n: The power spectrum ratio of overlapping protein(s) to the target protein. 
   - resolution: Only use this range of resolution for computing, e.g. 8-400 Å. 
   - Diameter: Similar to box size in RELION. Unit: pixel. 
   - Score_threshold: Particles with score lower than this will be removed. 
   - Window_size: The length of the search window, which should be multiples of 32, such as 480 and 512. 
   - Overlap: The overlapping length of neighboring search windows. It should be larger than the diameter of the target protein. 
   - Phase_flip: Whether to do phase flipping. 
   - Invert: Whether to invert contrast. 
   - Norm_type: Test function. Use 1 for now. 
7. Use the following command to remove repeated particles and convert LST file to STAR  file  (can  be  recognized  by  RELION-3.0). Taking  4  pixels  as  the  center distance and 8 degrees as the angle distance for instance,   
   `postprocess.py Output.lst 20 4 8 2 1.36 micrographs/`   
   The output file is `Output_merge.star`
8. (Optional) Convert the format of the output STAR file so that it can be recognized by RELION-3.1 (and above):   
   `relion30_to_31.py Output_merge.star`  
   The output file is `Output_merge_31.star` 
9. Use RELION to do Particle Extraction, 3D classification, and so on.
