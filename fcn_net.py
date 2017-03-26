# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 10:54:46 2017

@author: lhd
"""


import tensorflow as tf
import numpy as np

class FcnNet(object):
  
  def __init__(self, x ,
               weights_path = 'DEFAULT'):
    
    # Parse input arguments into class variables
    self.X = x
    
    # Call the create function to build the computational graph of AlexNet
    self.create()
    
  def create(self):
    
    # 1st Layer: Conv (w ReLu) -> Pool 
    conv1 = conv(self.X, 9, 9, 36, 1, 1, padding = 'SAME', name = 'conv1')
    pool1 = max_pool(conv1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')
    
    # 2nd Layer: Conv (w ReLu) -> Pool 
    conv2 = conv(pool1, 7, 7, 72, 1, 1, padding = 'SAME', name = 'conv2')
    pool2 = max_pool(conv2, 3, 3, 2, 2, padding = 'VALID', name ='pool2')
   
    
    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(pool2, 7, 7, 36, 1, 1, padding = 'SAME', name = 'conv3')
    
    # 4th Layer: Conv (w ReLu) 
    conv4 = conv(conv3, 7, 7, 24, 1, 1, padding = 'SAME', name = 'conv4')
    
    # 5th Layer: Conv (w ReLu) 
    conv5 = conv(conv4, 7, 7, 16, 1, 1, padding = 'SAME', name = 'conv5')
    
    
    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    conv6 = conv(conv5, 1, 1, 1, 1, 1, padding = 'SAME', name = 'conv5')
    
    # output
    self.output = conv6

"""
return sum(sum(row,column))
""" 

def integrator(self):
    
    self.integrator = tf.reduce_sum(self.output)
    
  
"""
Predefine all necessary layer for the AlexNet
""" 
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):  
             
  # Get number of input channels
  input_channels = int(x.get_shape()[-1])
  
  # Create lambda function for the convolution
  convolve = lambda i, k: tf.nn.conv2d(i, k, 
                                       strides = [1, stride_y, stride_x, 1],
                                       padding = padding)
  
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases of the conv layer
    #weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels/groups, num_filters])
    #biases = tf.get_variable('biases', shape = [num_filters])  
    weights = tf.Variable(tf.truncated_normal([filter_height, filter_width, input_channels, num_filters], dtype=tf.float32,
                                              stddev=1e-2), name='weights')
    biases = tf.Variable(tf.constant(0.0, shape=[num_filters], dtype=tf.float32),
                         trainable=True, name='biases')                                         
    
   
    conv = convolve(x, weights)
          
    # Add biases 
    bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
    # Apply relu function
    relu = tf.nn.relu(bias, name = scope.name)
        
    return relu
  
def fc(x, num_in, num_out, name, relu = True):
  with tf.variable_scope(name) as scope:
    
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)
    
    # Matrix multiply weights and inputs and add bias
    act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    
    if relu == True:
      # Apply ReLu non linearity
      relu = tf.nn.relu(act)      
      return relu
    else:
      return act
    

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='VALID'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)