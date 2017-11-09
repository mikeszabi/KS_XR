# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 12:31:50 2017

@author: SzMike
"""

import xml.etree.ElementTree as ET


class cfg:
    
    def __init__(self):
        # reading parameters
        tree = ET.parse('config.xml')
        self.params={}
        self.params['pixpermm'] = tree.find('params/pixpermm').text 
        self.params['smallsize'] = tree.find('params/smallsize').text