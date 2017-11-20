# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:25:46 2017

@author: SzMike
"""

import numpy as np
import cv2


mser = cv2.MSER_create()
mser.setMinArea(50)
mser.setMaxArea(100000)

## Do mser detection, get the coodinates and bboxes
coordinates, bboxes = mser.detectRegions(im_adj)

## Filter the coordinates
vis = im_orig.copy()
coords = []
for bbox in bboxes:
    cv2.rectangle(vis,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color=1)
 