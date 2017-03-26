# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 10:49:55 2017

@author: lhd
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

from datetime import datetime
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from fcn_net import FcnNet

from mall_input import ImageDataGenerator

"""
Configuration settings
"""

# Path to the textfiles for the trainings and validation set
train_file = 'train.txt'
test_file = 'val.txt'

batch_size = 1
train_image_size = 1200
test_image_size = 800

image_height = 480
image_width = 640
output_height = image_height/4
output_width = image_width/4

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./fcn_net/mall_dataset/file"
checkpoint_path = "./fcn_net/mall_dataset/checkpoint"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)
if not os.path.isdir(filewriter_path): os.mkdir(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = ImageDataGenerator(train_file, 
                                     horizontal_flip = False, shuffle = False)
test_generator = ImageDataGenerator(test_file, shuffle = False) 

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(test_generator.data_size / batch_size).astype(np.int16)

def train():
  # TF placeholder for graph input and output
  x = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 3])
  y = tf.placeholder(tf.float32, [batch_size, output_height, output_width])
  # Initialize model
  model = FcnNet(x)

  # Link variable to model output
  output = model.fc8
  
  loss = tf.reduce_mean(tf.square(y - output))
  
  with tf.name_scope("crowd_estimation"):
      prediction = model.integrator

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  global_step = tf.Variable(0, dtype=tf.float32)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      1e-6,                # Base learning rate.
      global_step,         # Current index into the dataset.
      1e6,                 # Decay step.
      0.1,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=global_step)

  
  # Predictions for the current training minibatch.
#  train_prediction = tf.nn.softmax(logits)
#
#  # Predictions for the test and validation, which we'll compute less often.
#  eval_prediction = tf.nn.softmax(model(eval_data))
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.global_variables_initializer().run()
    print('Initialized!')
    # Loop through training steps.
    iteration_size = 2e6
    train_size = train_image_size
    num_epochs = iteration_size/train_size
    BATCH_SIZE = batch_size
    
    
    saver_num = 1
     # Loop over number of epochs
    for epoch in range(num_epochs):
        
        print("{} Epoch number: {}".format(datetime.now(), epoch+1))
        
        step = 1
        
        while step < train_batches_per_epoch:         
            # Get a batch of images and labels
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {x: batch_xs,
                         y: batch_ys}
            # Run the optimizer to update weights.
            sess.run(optimizer, feed_dict=feed_dict)
            step += 1
    
        #After one epoch
        #Calculate the MAE and MSE
        mae = 0.
        mse = 0.
        for _ in range(test_image_size):
            batch_tx, batch_ty = test_generator.next_batch(batch_size)
            pred = sess.run(prediction, feed_dict={x: batch_tx})
            mae += np.abs(pred - batch_ty)
            mse += np.square(pred - batch_ty)
            
        mae /= 2*test_image_size
        mse /= 2*test_image_size
    
        print("{} Test Accuracy, MAE = {:.4f}, MSE = {:.4f}".format(datetime.now(), mae, mse))
        
        if global_step % 100000 == 0 :
            
            print("{} Saving checkpoint of model...".format(datetime.now()))              
            #save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_'+str(saver_num)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)  
            saver_num = saver_num + 1
            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
            
            
if __name__ == '__main__':
    train()            
            
            
            
            
            
            
            