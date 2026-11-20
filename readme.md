# Installation

Clone the repository or download tarball from <https://github.com/ZHANGXinzheng-LAB/isSPA/tags> and extract
```bash
make install -j8
pip3 install argparse numpy seaborn starfile mrcfile
```

Optional: add the absolute path of `isSPA/scripts` to the `PATH` environment variables. For example,
```bash
export PATH=/home/user/Software/isSPA/scripts:$PATH
```

# Manual  

0. It is highlt recommended to run isSPA inside a RELION project directory,  
1. Use RELION to do Import and CTF estimation, generating `micrographs_ctf.star` file. Here we have 20 micrographs as a test dataset. The pixel size is 1.36. Voltage is 300 kV. Amplitude contrast is 0.07. 
2. Use the following command to bin all the micrographs and store them in another directory, converting STAR file to LST file which can be read by isSPA. Taking bin2 for example,   
   `preprocess.py micrographs/ 2 1.36 CtfFind/job002/micrographs_ctf.star`   
   The binned micrographs are stored in `./micrographs/bin2/`
3. If needed, adjust the pixel size of the 3D template:   
   `relion_image_handler --i 3D_template.mrc --o 3D_template_rescaled.mrc --angpix 1.36 --rescale_angpix 2.72`
4. Generate quasi-evenly distributed orientations for projection. Here we generate 3-degree separated orientations and split into two files, 
   `/fast41a/zhuoer/github/isSPA/scripts/angle_step_generator.py -d 3 -N 2`
5. Use RELION to generate 2D projections of the 3D template,   
   `for i in {0..1}; do relion_project --i 3D_template.mrc --o 2D_templates_part$i --ang phi3.0_theta3.0_part$i.star; done`
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
