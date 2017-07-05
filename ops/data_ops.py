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

def preprocess(image):
    with tf.name_scope('preprocess'):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope('deprocess'):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope('preprocess_lab'):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope('deprocess_lab'):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
        #return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=2)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb



def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


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
      input_images = tf.image.resize_images(inputs, [crop_size, crop_size], method=tf.image.ResizeMethod.AREA)
      target_images = tf.image.resize_images(targets, [crop_size, crop_size], method=tf.image.ResizeMethod.AREA)

   inputsA_batch, inputsB_batch = tf.train.batch([inputsA, inputsB], batch_size=batch_size)

   return Data(
      inputsA=inputsA_batch,
      inputsB=inputsB_batch,
      count=len(imageA_paths)+len(imageB_paths),
   )
