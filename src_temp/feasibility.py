# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:13:31 2017

@author: SzMike
"""
import os
import skimage.io as io
from skimage import filters 
from skimage import exposure 
from skimage import restoration
from skimage import morphology
from skimage import segmentation
import matplotlib.pyplot as plt


%matplotlib qt5


row=df_4.iloc[0]

file_selected=os.path.join(r'e:\OneDrive\KS-XR\X-ray képek\Test\roi',
#                           row['Date'].replace('.',''),
                           str(row['Rotor_ID'])+'-'+row['Orientation']+'.jpg')

im_orig=io.imread(file_selected)

im_adj=exposure.rescale_intensity(im_orig,in_range=(100, 200))

im_bilat=restoration.denoise_bilateral(im_adj,multichannel=False,sigma_spatial=2)
im_frangi=filters.frangi(im_bilat,scale_range=(1, 5),scale_step=1)
im_laplace=filters.laplace(im_bilat,10)
#im_hessian=filters.hessian(im_bilat,scale_range=(1, 5),scale_step=1)
im_ent = filters.rank.entropy(im_bilat, morphology.disk(5))

im_mask = segmentation.felzenszwalb(im_adj, scale=200, sigma=2,min_size=10)

im_bound=segmentation.mark_boundaries(im_orig,im_mask)

# ToDO: Watershed using DoG centers

segments = segmentation.slic(im_adj, n_segments=200, compactness=1)
im_bound=segmentation.mark_boundaries(im_orig,segments)


plt.imshow(im_mask,cmap='gray')