# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:35:40 2017

@author: SzMike
"""

import os
import numpy as np
from skimage import morphology 
from skimage import segmentation 
from segmentations import segment_global_kmean
from skimage import measure 
from matplotlib import pyplot as plt
import matplotlib.patches as patches


from image_helper import mask_overlay



def get_roi(im,n_clusters=5,geo_bb=(0.28,0.83,0.35,0.65),vis_diag=False):
    center, label = segment_global_kmean(im,n_clusters=n_clusters,vis_diag=vis_diag)

    roi_mask=label>-1
    
    for i,c in enumerate(center):
        if c<0.1 or c>0.9:
            roi_mask[label==i]=0

    roi_mask = morphology.binary_opening(roi_mask, morphology.disk(3))      
    roi_mask = morphology.binary_closing(roi_mask, morphology.disk(5))

    #mask_overlay(im,roi_mask,0.5,ch=1,sbs=True,vis_diag=True)
    
    # use geometrical constraints
    n_row,n_col=roi_mask.shape
    
    geo_mask=np.zeros(roi_mask.shape)
    geo_mask[int(geo_bb[0]*n_row):int(geo_bb[1]*n_row),int(geo_bb[2]*n_col):int(geo_bb[3]*n_col)]=1
    
    #geo_masked=mask_overlay(im,geo_mask,0.3,ch=1,sbs=True,vis_diag=True)
    
    roi_mask=np.logical_and(roi_mask,geo_mask)
    roi_mask = morphology.binary_opening(roi_mask, morphology.disk(3))      

    if vis_diag:
        geo_masked=mask_overlay(im,geo_mask,0.3,ch=1,sbs=True,vis_diag=False)
        mask_overlay(geo_masked,morphology.binary_dilation(segmentation.find_boundaries(roi_mask),morphology.disk(3)),0.7,ch=2,sbs=False,vis_diag=True)
        mask_overlay(geo_masked,roi_mask,0.7,ch=2,sbs=False,vis_diag=True)

    return roi_mask

def crop_roi(im,roi_mask,pad_rate=0.5,save_file=None,vis_diag=False):

    label_im=measure.label(roi_mask)
    props = measure.regionprops(label_im)
    
    areas = [prop.area for prop in props]    
    l = [prop.major_axis_length  for prop in props]  
    bb=None
    im_cropped=None
    if areas:    
        if max(l)>max(im.shape[0],im.shape[1])*0.2:
            # ToDo: find area with largest edges1
            prop_large = props[np.argmax(areas)]       
            bb=prop_large.bbox
    if bb:
        dx=2*int((1+pad_rate)*(bb[3]-bb[1])/2)
        dy=bb[2]-bb[0]
        o=(int((bb[3]+bb[1])/2),int((bb[2]+bb[0])/2)) # x,y

        im_cropped = im[bb[0]:bb[2], o[0]-int(dx/2):o[0]+int(dx/2)]
        
        
        if save_file:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
            fig.suptitle(os.path.basename(save_file))
        
            ax1.imshow(im,cmap='gray')
            #ax1.axis('off')
       
            ax1.add_patch(patches.Rectangle(
                        (o[0]-int(dx/2), bb[0]),   # (x,y)
                            dx,        # width
                            dy,       # height
                            fill=False,color='Red'))
            
            ax2.imshow(im_cropped,cmap='gray')
            #ax2.axis('off')
            
            
            fig.savefig(save_file)
            plt.close('all')
    return im_cropped
        
        