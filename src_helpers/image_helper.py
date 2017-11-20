# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:57:09 2017

@author: SzMike
"""

import warnings
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rescale
from skimage.exposure import cumulative_distribution, adjust_gamma
from skimage import filters, img_as_ubyte, morphology
from mpl_toolkits.axes_grid1 import make_axes_locatable

# colorhist  - works for grayscale and color images
def gray_hist(im,vis_diag=False,nbins=256,mask=None,fig=''):
    assert im.ndim==2, 'Not 2channel image'
    assert im.dtype=='float64', 'Not float64 image'

    if mask is None:
        im_masked=im
    else:
        im_masked=im.flat[mask.flatten()>0]
            
    h, b=np.histogram(im_masked*255,range=(-1,255),bins=nbins, density=True)
    if vis_diag:
        fh=plt.figure(fig+'_histogram')
        ax=fh.add_subplot(111)
        ax.bar(b[1:],h)
        ax.set_xlim([0,255])
        #ax.set_ylim([0,0.1])
    return h
    
def mask_overlay(im,mask,alpha=0.5,ch=1,sbs=False,ax=None, vis_diag=False,fig=''):
# mask is 2D binary
# image can be 1 or 3 channel
# ch : rgb -> 012
# sbs: side by side
# http://stackoverflow.com/questions/9193603/applying-a-coloured-overlay-to-an-image-in-either-pil-or-imagemagik
    if ch>2:
        ch=1

    mask_tmp=np.empty(mask.shape+(3,), dtype='uint8')   
    mask_tmp.fill(0)
    mask_tmp[:,:,ch]=mask
    
    if im.ndim==2:
        im_3=np.matlib.repeat(np.expand_dims(im,2),3,2)
    else:
        im_3=im
        
    im_overlay=np.add(alpha*mask_tmp,(1-alpha)*im_3).astype(im.dtype)
    
    if vis_diag:
        if sbs:
            both = np.hstack((im_3,im_overlay))
        else:
            both=im_overlay
        if ax is None:
            fi=plt.figure(fig+'_overlayed')
            ax=fi.add_subplot(111)
        ax.imshow(both)
    return im_overlay

#def overlayImage(im, mask, col, alpha,ax=None,fig='',vis_diag=False):
#    assert im.ndim==3, 'Not 3channel image'
#    assert im.dtype=='uint8', 'Not uint8'
#    maskRGB = np.tile(mask[..., np.newaxis]>0, 3)
#    untocuhed = (maskRGB == False) * im
#    overlayComponent = 255* alpha * np.array(col) * maskRGB
#    origImageComponent = (1 - alpha) * maskRGB * im
#    with warnings.catch_warnings():
#        warnings.simplefilter("ignore")
#        im_overlay=img_as_ubyte((untocuhed + overlayComponent + origImageComponent)/255)
#    if vis_diag:
#        if ax is None:
#            fi=plt.figure(fig+'_overlayed')
#            ax=fi.add_subplot(111)
#        ax.imshow(im_overlay)
#    return im_overlay
    
def normalize(im,vis_diag=False,ax=None,fig=''):
    # normalize intensity image
    assert im.ndim==2, 'Not 1channel image'
    cdf, bins=cumulative_distribution(im, nbins=256)
    minI=bins[np.argwhere(cdf>0.01)[0,0]]
    maxI=bins[np.argwhere(cdf>0.99)[0,0]]
    im_norm=im.copy()
    im_norm[im_norm<minI]=minI
    im_norm[im_norm>maxI]=maxI
    im_norm=(im_norm-minI)/(maxI-minI)
    im_norm=(255*im_norm).astype('uint8')       
    if vis_diag:
        if ax is None:
            fi=plt.figure(fig+'_normalized')
            ax=fi.add_subplot(111)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            i=ax.imshow(im_norm,cmap='jet')
            fi.colorbar(i, cax=cax, orientation='vertical')
        else:
            ax.imshow(im_norm,cmap='jet')
        plt.show()
    return im_norm       
    
def get_gradient_magnitude(im):
    #Get magnitude of gradient for given image"
    assert len(im.shape)==2, "Not 2D image"
    mag = filters.scharr(im)
    return mag



def imRescaleMaxDim(im, maxDim, boUpscale = False, interpolation = 1):
    scale = 1.0 * maxDim / max(im.shape[:2])
    if scale < 1  or boUpscale:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            im = img_as_ubyte(rescale(im, scale, order=interpolation))
    else:
        scale = 1.0
    return im, scale

#def plotShapes(im, shapelist, detect_shapes='ALL',color='g', text='ALL', marker='.',ha='center',va='center',fig=None):
#    if fig is None:
#        fig=plt.figure('shapes image',figsize=(20,20))
#    axs=fig.add_subplot(111)
#    axs.imshow(im)  
#    for shape in shapelist:
#        pts=shape[2]
#        if type(pts) is list:
#            pts=np.asarray(pts)
#            pts=np.concatenate((pts,np.reshape(pts[0,:],(1,2))),axis=0)
#        if detect_shapes=='ALL' or shape[0] in detect_shapes:           
#            axs.plot(pts[:,0], pts[:,1], markersize =10, color=color, marker=marker)
#        if text=='ALL' or shape[0] in text:   
#            axs.annotate(shape[0],size=20,xy=(pts[0,0], pts[0, 1]),ha=ha,va=va,\
#                         bbox=dict(boxstyle="round", fc="White", ec=color, lw=1),color=color)
#    return fig

def adjust_gamma_gray(im,norm=128,med=128):
    assert im.ndim==2, "Not 2D image"
    im_adjust=im.copy()
    gamma=np.log(255-norm)/np.log(255-med)
    gain=min(255/im.max(),norm/np.power(med,gamma))
    im_adjust=adjust_gamma(im,gamma=np.mean(gamma),gain=np.mean(gain))
    return im_adjust


    
def histogram_similarity(hist, reference_hist):
   
    # Compute Chi squared distance metric: sum((X-Y)^2 / (X+Y));
    # a measure of distance between histograms
    X = hist
    Y = reference_hist

    num = (X - Y) ** 2
    denom = X + Y
    denom[denom == 0] = np.infty
    frac = num / denom

    chi_sqr = 0.5 * np.sum(frac, axis=0)

    # Generate a similarity measure. It needs to be low when distance is high
    # and high when distance is low; taking the reciprocal will do this.
    # Chi squared will always be >= 0, add small value to prevent divide by 0.
    similarity = 1 / (chi_sqr + 1.0e-4)

    return similarity

def walklevel(root_dir, level=1):
    root_dir = root_dir.rstrip(os.path.sep)
    assert os.path.isdir(root_dir)
    num_sep = root_dir.count(os.path.sep)
    for root, dirs, files in os.walk(root_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            
def imagelist_in_depth(image_dir,level=1):
    image_list_indir=[]
    included_extenstions = ['*.jpg', '*.bmp', '*.png', '*.gif']
    image_list_indir = []
    for root, dirs, files in walklevel(image_dir, level=level):
        for ext in included_extenstions:
            image_list_indir.extend(glob.glob(os.path.join(root, ext)))
    return image_list_indir