# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:56:02 2017

@author: SzMike
"""

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import matplotlib.pyplot as plt


def blob_detections(image):
    
    image_gray = rgb2gray(image).astype('float64')/255
    
    blobs_log = blob_log(1-image_gray, max_sigma=30, num_sigma=10, threshold=0.05)
    
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    
    blobs_dog = blob_dog(1-image_gray, max_sigma=30, threshold=0.05)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    
    blobs_doh = blob_doh(1-image_gray, max_sigma=30, threshold=.005)
    
    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()
    
    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image, interpolation='nearest', cmap='gray')
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()
    
    plt.tight_layout()
    plt.show()