# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 20:21:20 2017

@author: Szabolcs Mike
Create image db file
Uses class_map
"""

import csv
import numpy as np
import pandas as pd
import os
from src_train.train_config import train_params
import src_tools.file_helper as fh

#==============================================================================
# SET THESE PARAMETERS!
#==============================================================================

training_id='20171130'
curdb_dir='roi'
data_dir=r'c:\Users\picturio\OneDrive\KS-XR\Images'
count_threshold = 0


#==============================================================================
# RUN CONFIG
#==============================================================================

cfg=train_params(data_dir,curdb_dir=curdb_dir,training_id=training_id)


"""
Class names from folder names
"""
image_list=fh.imagelist_in_depth(cfg.curdb_dir,level=2)

df=pd.read_csv(cfg.base_db_file,delimiter=';')
fnames=[]
class_ids=[]
for f in image_list:
    file_name=os.path.splitext(os.path.basename(f))[0]
    Rotor_ID,Orientation=file_name.split('-')
    row=df[(df['Rotor_ID']==int(Rotor_ID)) & (df['Orientation']==Orientation)]
    if row.size>0:
        is_faulty=row['Faulty'].values[0]
        class_id=row['Classification'].values[0]
        if is_faulty==0:
            class_id=1
        fnames.append(f)
        class_ids.append(class_id)
    
df_db = pd.DataFrame(data={'Filename':fnames,'Class name':class_ids})


classes_count=df_db['Class name'].value_counts()

df_db.to_csv(cfg.db_file,sep=';',index=None)
