import numpy as np
import cv2
import scipy.io as spio
"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""

class ImageDataGenerator:
    def __init__(self, data_path, horizontal_flip=False, shuffle=False, 
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227)):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        
        self.read_data(data_path)
        
        if self.shuffle:
            self.shuffle_data()

    def read_data(self,data_path):
        """
        Scan the image file and get the image paths and labels
        """
        with open(data_path) as f:
            lines = f.readlines()
            self.images = []
            self.dmap = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.dmap.append(items[1])
                self.labels.append(int(items[2]))
            
            #store total number of data
            self.data_size = len(self.labels)
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.images.copy()
        labels = self.labels.copy()
        dmap = self.dmap.copy()
        self.images = []
        self.dmap = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.dmap.append(dmap[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size, resize = False):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        paths_d = self.dmap[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]
        
        #update pointer
        self.pointer += batch_size
        
        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        dmap = np.ndarray([batch_size, self.scale_size[0]/4, self.scale_size[1]/4, 3 ])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])
            dm = spio.loadmat(paths_d[i])
            
            #flip image at random if flag is selected
            if self.horizontal_flip and np.random.random() < 0.5:
                img = cv2.flip(img, 1)
            
            #rescale image
            if resize == True:
                img = cv2.resize(img, (self.scale_size[0], self.scale_size[0]))
            
            img = img.astype(np.float32)
            
            #subtract mean
            #img -= self.mean
                                                                 
            images[i] = img
            dmap[i] = dm['d_map'].astype(np.float32)
        # Expand labels to one hot encoding
#        one_hot_labels = np.zeros((batch_size, self.n_classes))
#        for i in range(len(labels)):
#            one_hot_labels[i][labels[i]] = 1

        #return array of images and labels
        return images, dmap, labels