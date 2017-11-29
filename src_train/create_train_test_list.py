# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 13:48:16 2017

@author: SzMike


Creates train and test lists from the images 
Each classes used have to have enough observations (min_obs)
creates type_dict
"""

#training_id='20171120-All'

import csv
import pandas as pd
import os
import numpy as np
import collections

from src_train.train_config import train_params
#imp.reload(sys.modules['train_params'])

def keysWithValue(aDict, target):
    return sorted(key for key, value in aDict.items() if target == value)

# do startified random split in the data
def get_stratified_train_test_inds(y,train_proportion=0.75):
    '''Generates indices, making random stratified split into training set and testing sets
    with proportions train_proportion and (1-train_proportion) of initial sample.
    y is any iterable indicating classes of each observation in the sample.
    Initial proportions of classes inside training and 
    testing sets are preserved (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return np.where(train_inds)[0],np.where(test_inds)[0]



#==============================================================================
# SET THESE PARAMETERS!
#==============================================================================
curdb_dir='roi'
data_dir=r'e:\OneDrive\KS-XR\X-ray képek'

#==============================================================================
# RUN CONFIG
#==============================================================================


cfg=train_params(data_dir,curdb_dir=curdb_dir,training_id=training_id)

"""
Read data description file
"""

df_db = pd.read_csv(cfg.db_file,delimiter=';')

"""
Select classes to process
"""

df_filtered=df_db.copy()

  
df_labeled=df_db[['Filename']].copy()
df_labeled['category']=df_db[['Class name']]
df_labeled.columns=['image','category']


"""
Spit to test and train sest
"""
train_inds,test_inds = get_stratified_train_test_inds(df_labeled['category'], cfg.trainRatio)
np.random.shuffle(train_inds)
np.random.shuffle(test_inds)
df_train_image=df_labeled.iloc[train_inds]
df_test_image=df_labeled.iloc[test_inds]



"""
Do some stats
"""
num_classes=len(df_labeled['category'].value_counts())

classes_count_train=df_train_image['category'].value_counts()
print(len(df_train_image))
classes_count_test=df_test_image['category'].value_counts()
print(len(df_test_image))
# number of classes
print(num_classes)

"""
Write train and test list
"""
df_train_image.to_csv(cfg.train_image_list_file,sep=';',index=None)
df_test_image.to_csv(cfg.test_image_list_file,sep=';',index=None)

