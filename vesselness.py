# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:18:18 2017

@author: SzMike
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import imp

#imp.reload(sys.modules['image_helper'])

import os
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import numpy as np

%matplotlib qt5


from skimage import segmentation
from skimage import transform
from skimage import feature
from skimage import filters


from file_helper import imagelist_in_depth

from image_helper import gray_hist, mask_overlay, get_gradient_magnitude

%matplotlib qt5

roi_dir=r'E:\OneDrive\KS-XR\X-ray képek\Test\roi'

image_file_list=imagelist_in_depth(roi_dir)

for i, image_file in enumerate(image_file_list):
    i=312
    print(i)
    im_orig=io.imread(image_file_list[i])
    
    fig, ax = plt.subplots(ncols=3, subplot_kw={'adjustable': 'box-forced'})
    
    ax[0].imshow(im_orig, cmap=plt.cm.gray)
    ax[0].set_title('Original image')
    
    ax[1].imshow(filters.frangi(im_orig,scale_range=(1, 5),scale_step=1)>0.0000005, cmap=plt.cm.gray)
    ax[1].set_title('Frangi filter result')
    
#    ax[2].imshow(filters.hessian(im_orig,scale_range=(1, 5),beta1=10), cmap=plt.cm.gray)
#    ax[2].set_title('Hybrid Hessian filter result')
    
    # felzenszwalb
    im_mask = segmentation.felzenszwalb(im_orig, scale=100, sigma=2,min_size=10)
    # 500-ra . scale=1000, sigma=1.5,min_size=10000
    ax[2].imshow(im_mask)
    ax[2].set_title('Felzenswalb')
    
    for a in ax:
        a.axis('off')
    
    plt.tight_layout()