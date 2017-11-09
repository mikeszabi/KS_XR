# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import imp

#imp.reload(sys.modules['roi'])
from roi import get_roi, crop_roi

import os
import skimage.io as io
io.use_plugin('pil') # Use all capabilities provided by PIL
from PIL import Image

#imp.reload(sys.modules['cfg'])
from cfg import cfg

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import numpy as np

from skimage import color
from skimage import transform
from image_helper import mask_overlay
from file_helper import dirlist_onelevel




#from skimage import img_as_ubyte, img_as_float
#from skimage.transform import rescale
#from cntk_helpers import *

#%matplotlib qt5


# =============================================================================
# PARAMETERS
# =============================================================================

p=cfg()

size_small=int(p.params['smallsize'])
ppmm=int(p.params['pixpermm'])

data_dir=r'E:\OneDrive\KS-XR\X-ray képek\201710'
save_dir=r'E:\OneDrive\KS-XR\X-ray képek\Test\roi'
save_dir_temp=r'E:\OneDrive\KS-XR\X-ray képek\Test\roi_crop'


measure_ids=dirlist_onelevel(data_dir)

ims=[]


for measure_id in measure_ids:

    # measure_id=measure_ids[0]

    measure_dir=os.path.join(data_dir,measure_id)

    coll = io.ImageCollection(measure_dir + '\\*.jpg')

for i, im_orig in enumerate(coll):
    
    im_file=r'E:\OneDrive\KS-XR\X-ray képek\Test\roi_problems\2351-G.jpg'
    #im_orig=io.imread(im_file)
    
    im_file=coll.files[i]

# small size
    im=transform.resize(im_orig, (size_small, size_small), mode='reflect')

    roi_mask=get_roi(im,n_clusters=5,vis_diag=False)
    
# visualize croping    
    roi_masked=255*mask_overlay(im,roi_mask,0.5,ch=1,sbs=True,vis_diag=False)
    
    save_file_temp=os.path.join(save_dir_temp,os.path.basename(im_file))
   
    im_cropped=crop_roi(roi_masked.astype('uint8'),roi_mask,pad_rate=0.5,save_file=save_file_temp,vis_diag=False)


# crop original size
    save_file=os.path.join(save_dir,os.path.basename(im_file))


    roi_mask_orig=transform.resize(roi_mask, im_orig.shape, mode='reflect')
    im_cropped=crop_roi(im_orig,roi_mask_orig,pad_rate=0.5)

    img = Image.fromarray(im_cropped.astype('uint8'))
    img.save(save_file)
    