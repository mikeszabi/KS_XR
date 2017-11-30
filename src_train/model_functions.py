# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 10:44:25 2017

@author: picturio
"""
from cntk import relu
from cntk.layers import Convolution, MaxPooling, Dropout, Dense, For, default_options
from cntk.initializer import glorot_uniform

def create_shallow_model(input, out_dims):
    
    convolutional_layer_1_1  = Convolution((7,7), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(input)
    convolutional_layer_1_2  = Convolution((25,25), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(convolutional_layer_1_1)

    pooling_layer_1  = MaxPooling((25,25), strides=(5,5))(convolutional_layer_1_2 )
    
    convolutional_layer_2_1 = Convolution((3,3), 32, init=glorot_uniform(), activation=relu, pad=True, strides=(1,1))(pooling_layer_1)
    pooling_layer_2 = MaxPooling((2,2), strides=(2,2))(convolutional_layer_2_1)
 
    fully_connected_layer_1  = Dense(512, init=glorot_uniform())(pooling_layer_2)   
    fully_connected_layer_2  = Dense(128, init=glorot_uniform())(fully_connected_layer_1)
    dropout_layer = Dropout(0.5)(fully_connected_layer_2)

    output_layer = Dense(out_dims, init=glorot_uniform(), activation=None)(dropout_layer)
    
    return output_layer


