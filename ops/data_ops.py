'''

Operations used for data management

MASSIVE help from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

'''

from __future__ import division
from __future__ import absolute_import

from scipy import misc
from skimage import color
import collections
import tensorflow as tf
import numpy as np
import math
import time
import random
import glob
import os
import fnmatch
import cPickle as pickle

Data = collections.namedtuple('trainData', 'inputsA, inputsB, count')

# [0, 255] => [-1, 1]
def preprocess(image):
   with tf.name_scope('preprocess'):
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return (image/127.50)-1.0

# [-1, 1] => [0, 1]
def deprocess(image):
   with tf.name_scope('deprocess'):
      return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def getPaths(data_dir, gray_images=None, ext='jpg'):
   pattern   = '*.'+ext
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fname_ = os.path.join(d,filename)
            if gray_images is not None:
               if fname_ in gray_images:
                  continue
            image_paths.append(fname_)
   return image_paths


# TODO add in files to exclude (gray ones)
def loadData(dataset, batch_size, train=True):

   # data directory plus the dataset being used
   data_dir = '/mnt/data2/images/cyclegan/datasets/'+dataset+'/'

   # check to see if dataset actually exists
   if dataset is None or not os.path.exists(data_dir):
      raise Exception('data_dir '+data_dir+' does not exist')

   print 'Getting data...'

   if train:
      imageA_dir = data_dir+'trainA/'
      imageB_dir = data_dir+'trainB/'
   else:
      imageA_dir = data_dir+'testA/'
      imageB_dir = data_dir+'testB/'
      
   imageA_paths = getPaths(imageA_dir)
   imageB_paths = getPaths(imageB_dir)
   random.shuffle(imageA_paths)
   random.shuffle(imageB_paths)

   if len(imageA_paths) == 0:
      raise Exception('data_dir contains no image files')
   print 'Done!'

   print len(imageA_paths)+len(imageB_paths),'images!'
   decode = tf.image.decode_image
   
   with tf.name_scope('load_images'):

      # create a queue from the input paths
      pathA_queue = tf.train.string_input_producer(imageA_paths, shuffle=True)
      pathB_queue = tf.train.string_input_producer(imageB_paths, shuffle=True)

      # read in contents from the path
      reader = tf.WholeFileReader()
      pathsA, contentsA = reader.read(pathA_queue)
      pathsB, contentsB = reader.read(pathB_queue)
      
      # decode image, then convert to float32
      inputsA = tf.image.convert_image_dtype(decode(contentsA), dtype=tf.float32)
      inputsB = tf.image.convert_image_dtype(decode(contentsB), dtype=tf.float32)

      inputsA.set_shape([256, 256, 3])
      inputsB.set_shape([256, 256, 3])

   # randomly flip and rescale images -> increse image size by 20 pixels then crop it back to 256x256
   scale_size = 256+20
   crop_size  = 256
   seed = random.randint(0, 2**31 - 1) 
   def transform(image):
      r = image
      r = tf.image.random_flip_left_right(r, seed=seed)

      r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
      offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - crop_size + 1, seed=seed)), dtype=tf.int32)
      r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], crop_size, crop_size)
      return r

   if train:
      with tf.name_scope('inputsA'):
         inputsA = transform(inputsA)
      with tf.name_scope('inputsB'):
         inputsB = transform(inputsB)
   else:
      input_images = tf.image.resize_images(inputsA, [crop_size, crop_size], method=tf.image.ResizeMethod.AREA)
      target_images = tf.image.resize_images(inputsB, [crop_size, crop_size], method=tf.image.ResizeMethod.AREA)

   inputsA_batch, inputsB_batch = tf.train.batch([inputsA, inputsB], batch_size=batch_size)

   return Data(
      inputsA=inputsA_batch,
      inputsB=inputsB_batch,
      count=len(imageA_paths)+len(imageB_paths),
   )
