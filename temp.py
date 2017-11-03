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

%matplotlib qt5


# =============================================================================
# PARAMETERS
# =============================================================================

size_small=500

data_dir=r'd:\Projects\KS_XR\data\2-sarga'
measure_ids=dirlist_onelevel(data_dir)

edges=[]
hists=[]
ims=[]

for measure_id in measure_ids:

    #measure_id=measure_ids[0]

    measure_dir=os.path.join(data_dir,measure_id)

    image_file_list=imagelist_in_depth(measure_dir)
    
    for image_file in image_file_list:
       
        #image_file=image_file_list[0]
        
        im_orig=io.imread(image_file)
        im_small=transform.resize(im_orig, (size_small, size_small), mode='reflect')
        
        ims.append(im_small)
        
        
        h=gray_hist(im_small,vis_diag=False,nbins=128)        
        hists.append(h)
                
        ##
        edges1 = feature.canny(im_small, sigma=3, low_threshold=0.05, high_threshold=0.25, use_quantiles=False)
        edges.append(edges1)
        #plt.imshow(edges1)

# line detection??? 
        
im_avg=np.squeeze(np.mean(np.dstack(ims),axis=2)) 
hist_avg=np.squeeze(np.mean(np.dstack(hists),axis=2))   
edg_avg=np.squeeze(np.mean(np.dstack(edges),axis=2))  


   
# histogram
fh = plt.figure('segment')
ax0=fh.add_subplot(111)
ax0.bar(np.arange(0., 256, 2),hist_avg)


# edg_avg
mask_overlay(im_avg,edg_avg>0,0.5,ch=1,sbs=True,vis_diag=True)
mask_overlay(ims[i],edges[i]>0,0.5,ch=1,sbs=True,vis_diag=True)

# felzenswalb
fh = plt.figure('segment')
ax0=fh.add_subplot(121)
ax1=fh.add_subplot(122)

i=1
im_mask = segmentation.felzenszwalb(ims[i], scale=500, sigma=1.5,min_size=1000)
# 500-ra . scale=1000, sigma=1.5,min_size=10000
ax0.imshow(ims[i],cmap='gray')
ax1.imshow(im_mask)

mask_overlay(ims[i],segmentation.find_boundaries(im_mask),0.5,ch=1,sbs=True,vis_diag=True)

#plt.imshow(segmentation.find_boundaries(im_mask))


# global kmeans
i=0
center, label = segment_global_kmean(ims[i],n_clusters=10)
        
mask_overlay(im_small,label==3,0.3,ch=1,sbs=False,vis_diag=True)

# hough line

lines = transform.probabilistic_hough_line(edg_avg>0.5, threshold=0.8, line_length=50,line_gap=3)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(im_avg, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edg_avg>0.5, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edg_avg * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, edg_avg.shape[1]))
ax[2].set_ylim((edg_avg.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()
    a.set_adjustable('box-forced')

plt.tight_layout()
plt.show()

##
# porosity

from skimage.filters import gaussian

image_file=image_file_list[3]
im_orig=io.imread(image_file)
fh = plt.figure('segment')
ax0=fh.add_subplot(121)
ax1=fh.add_subplot(122)

ax0.imshow(im_orig,cmap='gray')

im_orig=gaussian(im_orig, sigma=2)
ax1.imshow(im_blur,cmap='gray')

##
# CLAHE
from skimage import exposure
from skimage import feature
from math import sqrt

im_clahe=exposure.equalize_adapthist(im_orig,kernel_size=(50,50))

ax1.imshow(im_orig,cmap='gray')

blobs_dog = feature.blob_dog(1-im_orig, max_sigma=20, threshold=0.1)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
for blob in blobs_dog:
    y, x, r = blob
    c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    ax1.add_patch(c)
ax1.set_axis_off()

plt.tight_layout()
plt.show()

##
from skimage.morphology import disk
from skimage.filters.rank import bottomhat, otsu, entropy


out = bottomhat(im_orig, disk(10))
ax1.imshow(out,cmap='gray')

##
ent = entropy(im_orig, disk(10))
ax1.imshow(ent>4.5,cmap='gray')

##
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

local_maxi = peak_local_max(1-im_orig, indices=False, footprint=np.ones((10, 10)))
ax1.imshow(local_maxi,cmap='gray')

markers = ndi.label(local_maxi)[0]
labels = watershed(im_orig, markers)
ax1.imshow(labels)

# felzenszwalb
im_mask = segmentation.felzenszwalb(im_orig, scale=50, sigma=1.5,min_size=50)
# 500-ra . scale=1000, sigma=1.5,min_size=10000
ax1.imshow(im_mask)