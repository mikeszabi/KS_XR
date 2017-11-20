# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 10:33:32 2017

@author: SzMike
"""
import warnings
import numpy as np;
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from src_helpers.image_helper import normalize


def segment_global_kmean(im,  init_centers='k-means++', n_clusters=3, vis_diag=False): 
    

    Z = im.reshape(-1,1)    
    Z = np.float32(Z)
                  

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, init=init_centers).fit(Z)
    
    center = kmeans.cluster_centers_
    label = kmeans.labels_
    #print(center)
 
    label_orig=np.resize(label,im.shape)
    
    normalize(label_orig,vis_diag=vis_diag,fig='labels')
   
    return center, label_orig

def center_diff_matrix(centers,metric='euclidean'):    
    return pairwise_distances(centers,metric='euclidean')