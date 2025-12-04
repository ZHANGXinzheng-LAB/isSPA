#!/usr/bin/env python3

import os

with open('./rm_files') as f:
    a = f.readlines()    
for i in a:
    file = i.strip().split('/')[-1]
    fname = f'../particles/{file}'
    os.remove(fname)