# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import imp

#imp.reload(sys.modules['roi'])
from roi import get_roi

import os
import skimage.io as io
io.use_plugin('pil') # Use all capabilities provided by PIL
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import numpy as np

from skimage import segmentation
from skimage import transform
from skimage import feature
from skimage import filters


from file_helper import dirlist_onelevel, imagelist_in_depth
from segmentations import segment_global_kmean
from image_helper import gray_hist, mask_overlay, get_gradient_magnitude

#from skimage import img_as_ubyte, img_as_float
#from skimage.transform import rescale
#from cntk_helpers import *

#%matplotlib qt5


# =============================================================================
# PARAMETERS
# =============================================================================

size_small=250

data_dir=r'E:\OneDrive\Kienle\X-ray képek\201710'
save_dir=r'D:\Projects\KS_XR\out'

measure_ids=dirlist_onelevel(data_dir)

ims=[]


for measure_id in measure_ids:

    # measure_id=measure_ids[0]

    measure_dir=os.path.join(data_dir,measure_id)

    coll = io.ImageCollection(measure_dir + '\\*.jpg')

#
#    image_file_list=imagelist_in_depth(measure_dir)
#    
#    for image_file in image_file_list:
    
    for im_orig in coll:
       
        #image_file=image_file_list[0]
        
#        im_orig=io.imread(image_file)
        im_small=transform.resize(im_orig, (size_small, size_small), mode='reflect')
        
        ims.append(im_small)
        
        
#        h=gray_hist(im_small,vis_diag=False,nbins=128)        
#        hists.append(h)
                
        ##
#        edges1 = feature.canny(im_small, sigma=3, low_threshold=0.05, high_threshold=0.25, use_quantiles=False)
#        edges.append(edges1)
        #plt.imshow(edges1)

for i in range(len(ims)):
    roi_mask=get_roi(ims[i],n_clusters=5,vis_diag=False)
    roi_masked=255*mask_overlay(ims[i],roi_mask,0.5,ch=1,sbs=True,vis_diag=False)

    img = Image.fromarray(roi_masked.astype('uint8'))
    img.save(os.path.join(save_dir,str(i)+'.jpg'))
    