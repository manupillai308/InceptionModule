#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

def inception(inputs, **kwargs):
    '''
    Inception Module
    Creates an inception module architecture with provided input.
    
    Args:
        input: input tensor
        kwargs: parameters to be passed on to convolutional layers
        
    NOTE:
        kwargs are directly passed on as a parameter to every convolutional layer, please avoid layer specific arguments
        like 'name'.
        
    Returns: 
        an inception layer/module
    '''
    with tf.variable_scope("layer1"):
        conv11 = tf.layers.conv2d(inputs=inputs,kernel_size=[1,1] ,strides=[1,1], padding="SAME", name="convolutional1_layer1",  **kwargs)
        conv12 = tf.layers.conv2d(inputs=inputs, kernel_size=[1,1],strides=[1,1], padding="SAME", name="convolutional2_layer1", **kwargs)
        pool11 = tf.layers.max_pooling2d(inputs=inputs, strides=[1,1], padding="SAME",pool_size=[3,3], name="convolutional3_layer1")
    
    with tf.variable_scope("layer2"):
        conv21 = tf.layers.conv2d(inputs=inputs, kernel_size=[1,1], strides=[1,1], padding="SAME", name="convolutional1_layer2", **kwargs)
        conv22 = tf.layers.conv2d(inputs=conv11, kernel_size=[3,3], strides=[1,1], padding="SAME", name="convolutional2_layer2",**kwargs)
        conv23 = tf.layers.conv2d(inputs=conv12, kernel_size=[5,5], strides=[1,1], padding="SAME", name="convolutional3_layer2", **kwargs)
        conv24 = tf.layers.conv2d(inputs=pool11, kernel_size=[1,1], strides=[1,1], padding="SAME", name="convolutional4_layer2", **kwargs)
    
    with tf.variable_scope("depth_concatenation"):
        depth_concat = tf.concat(values=[conv21, conv22, conv23, conv24], axis=3)
    
    return depth_concat

