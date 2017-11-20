# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:28:34 2017

@author: SzMike
"""
import os
import pandas as pd


data_dir=r'e:\OneDrive\KS-XR\X-ray képek\201710'

measure_file='measurements.csv'

df=pd.read_csv(os.path.join(data_dir,measure_file),delimiter=';')

ind_4=df.iloc(df['Classification']==4)


df_4=df[(df['Classification']==4) & (df['Faulty']==1)]
df_3=df[(df['Classification']==3) & (df['Faulty']==1)]
df_2=df[(df['Classification']==2) & (df['Faulty']==1)]





