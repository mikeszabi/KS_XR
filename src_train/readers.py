# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:34:22 2017

@author: picturio
"""
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs, transforms

#
# Define the reader for both training and evaluation action.
#


def create_reader(map_file, mean_file, train, image_height=800, image_width=150, num_channels=3, num_classes=32):
  
    # transformation pipeline for the features has crop only when training

    trs = []
    if train:
        trs += [
            transforms.crop(crop_type='center', aspect_ratio=0.1875, side_ratio=0.95, jitter_type='uniratio') # Horizontal flip enabled
        ]
    trs += [
        transforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
#        transforms.mean(mean_file)
    ]
    # deserializer
    image_source=ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=trs), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    ))
    return MinibatchSource(image_source)

