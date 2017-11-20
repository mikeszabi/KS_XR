# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 14:18:53 2017

@author: SzMike
"""
import sys
import imp

import os
import numpy as np
import skimage.io as io
io.use_plugin('pil') # Use all capabilities provided by PIL
from skimage import transform
import matplotlib.pyplot as plt

#imp.reload(sys.modules['cfg'])
from cfg import cfg

from src_helpers.file_helper import dirlist_onelevel
#imp.reload(sys.modules['src_tools.roi'])
from src_tools.roi import check_geometry

#%matplotlib qt5


# =============================================================================
# PARAMETERS
# =============================================================================
# check values for rotor part
check_vals={}

check_vals['area']=0  
check_vals['y_top']=0
check_vals['x_start']=0
check_vals['y_bottom']=0
check_vals['x_end']=0



p=cfg()

size_small=int(p.params['smallsize'])
ppmm=int(p.params['pixpermm'])

data_dir=r'E:\OneDrive\KS-XR\X-ray képek\201710'
save_dir_temp=r'E:\OneDrive\KS-XR\X-ray képek\Test\rotor_crop'


measure_ids=dirlist_onelevel(data_dir)

checks=[]

for measure_id in measure_ids:

    # measure_id=measure_ids[0]

    measure_dir=os.path.join(data_dir,measure_id)

    coll = io.ImageCollection(measure_dir + '\\*.jpg')

    for i, im_orig in enumerate(coll):
        
        im_file=r'E:\OneDrive\KS-XR\X-ray képek\Test\roi_problems\2357-B.jpg'
        
        im_file=coll.files[i]
        # im_orig=io.imread(im_file)
    
    # small size
        im=transform.resize(im_orig, (size_small, size_small), mode='reflect')
    
        
    # visualize croping    
        
        save_file_temp=os.path.join(save_dir_temp,os.path.basename(im_file))
       
        check=check_geometry(im,n_clusters=2,save_file=save_file_temp,vis_diag=False)
        checks.append(check)
 
area=[]
x_end=[]
x_start=[]
y_bottom=[]
y_top=[]    
for check in checks:
    area.append(check['area'])
    x_end.append(check['x_end'])
    x_start.append(check['x_start'])
    y_bottom.append(check['y_bottom'])
    y_top.append(check['y_top'])
    
plt.hist(np.asarray(x_end), bins='auto')