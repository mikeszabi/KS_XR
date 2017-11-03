# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:35:40 2017

@author: SzMike
"""


import numpy as np
from skimage import morphology 
from skimage import segmentation 

from image_helper import mask_overlay

from segmentations import segment_global_kmean


def get_roi(im,n_clusters=3,vis_diag=False):
    center, label = segment_global_kmean(im,n_clusters=n_clusters,vis_diag=False)

    roi_mask=label>0
    
    for i,c in enumerate(center):
        if c<0.2 or c>0.8:
            roi_mask[label==i]=0
      
    roi_mask = morphology.binary_opening(roi_mask, morphology.disk(10))
    #mask_overlay(im,roi_mask,0.5,ch=1,sbs=True,vis_diag=False)
    
    # use geometrical constraints
    n_row,n_col=roi_mask.shape
    
    geo_mask=np.zeros(roi_mask.shape)
    geo_mask[int(0.27*n_row):int(0.83*n_row),int(0.38*n_col):int(0.58*n_col)]=1
    
    #geo_masked=mask_overlay(im,geo_mask,0.3,ch=1,sbs=True,vis_diag=False)
    
    roi_mask=np.logical_and(roi_mask,geo_mask)

    if vis_diag:
        geo_masked=mask_overlay(im,geo_mask,0.3,ch=1,sbs=True,vis_diag=False)
        mask_overlay(geo_masked,morphology.binary_dilation(segmentation.find_boundaries(roi_mask),morphology.disk(3)),0.7,ch=2,sbs=False,vis_diag=True)

    return roi_mask