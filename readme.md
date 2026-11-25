# Installation

1. Clone the repository or download a tarball from <https://github.com/ZHANGXinzheng-LAB/isSPA/tags> and extract it.
```bash
make install -j8
```

2. The helper scripts for isSPA are written in Python 3, and require the following modules:
```bash
pip3 install argparse numpy seaborn starfile mrcfile
```

3. (Optional) Add the absolute path of `isSPA/scripts` to the `PATH` environment variable.
```bash
export PATH=/home/user/Software/isSPA/scripts:$PATH
```



# Manual  

0. It is highly recommended to run isSPA inside a RELION project directory.
1. Use RELION to do Import (and Motion correction) and CTF estimation, generating a `micrographs_ctf.star` file (such as job002 here).
2. Use the following command to convert the micrographs STAR file to an LST file that can be read by isSPA,   
   `preprocess.py CtfFind/job002/micrographs_ctf.star`   
3. Generate quasi-evenly distributed orientations for projection. Here we generate 3-degree-separated orientations and split them into two files,   
   `angle_step_generator.py -d 3 -N 2`
4. Use RELION to generate 2D projections of the 3D template,   
   `for i in {0..1}; do relion_project --i 3D_template.mrc --o 2D_templates_part$i --ang phi3.0_theta3.0_part$i.star; done`
5. Modify the `config` file and run isSPA (you may prepare a `config` for each micrograph LST file and each projection file),    
   `isSPA config` 
   - Pixel_size: The pixel size of input micrographs, or generated micrographs if you specify `--bin` in `preprocess.py`
   - Phi_step: Search step (in degrees) for angle phi, i.e. the in-plane rotation angle.
   - n: The power spectrum ratio of the overlapping protein(s) to the target protein. 
   - resolution: Only use this resolution range for computing, e.g. 6-400 Å. 
   - Diameter: Similar to box size in RELION. Unit: pixel. 
   - Score_threshold: Particles with score lower than this will be removed. 
   - Window_size: The length of the search window, which should be a multiple of 32, such as 480 and 512. 
   - Overlap: The overlapping length of neighboring search windows. It should be larger than the diameter of the target protein. 
   - Phase_flip: Whether to do phase flipping. 
   - Invert: Whether to invert contrast. 
6. Concatenate the picked particles into one LST file,   
   `cat pick_part*.lst > pick.lst`
7. Remove repeated particles and convert the particles LST file to a STAR file. Taking 8 pixels as the center distance and 12 degrees as the angle distance as an example,     
   `postprocess.py pick.lst 8 12 1.19 --score 6.6`   
8. Set `Refined particles STAR file:` to `pick_6.6_merge_31.star` in RELION Particle Extraction job.   
   If target protein contains symmetry, `relion_star_handler --remove_duplicates 8 --i Extract/job003/particles.star --o Extract/job003/particles_sym.star`
9. Run RELION Local 3D Refinement using `Extract/job003/particles.star` as input (or `Extract/job003/particles_sym.star` if symmetry). 
10. Calculate the phase residual of the refined particles and subset the high-scoring particles to remove model bias. 
    ```bash
    relion_project --i Refine3D/job004/run_class001.mrc --ctf --ang Refine3D/job004/run_data.star --o Refine3D/job004/run_proj
    score_particles.py -i Refine3D/job004/run_data.star -r Refine3D/job004/run_proj.mrcs
    explore_scores.py -i Refine3D/job004/run_data.npy
    subset_particles.py -i Refine3D/job004/run_data.star --thres 0.06
    ```
    Then use `Refine3D/job004/run_data_sub.star` as input to do 3D classification, Local 3D Refinement and so on.  
