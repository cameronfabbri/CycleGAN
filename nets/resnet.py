import tensorflow as tf
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def Rk(x, name, reuse=False):

   padded_x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')

   # layer 1
   conv1 = tf.layers.conv2d(padded_x, 128, 3, strides=1, name='g-'+name+'-1', padding='VALID', reuse=reuse)
   conv1 = instance_norm(conv1)
   conv1 = tf.nn.relu(conv1)

   padded_conv1 = tf.pad(conv1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
   
   # layer 2
   conv2 = tf.layers.conv2d(padded_conv1, 128, 3, strides=1, name='g-'+name+'-2', padding='VALID', reuse=reuse)
   conv2 = instance_norm(conv2)

   output = x+conv2
   #output = tf.nn.relu(output) # this may not be correct
   
   return output

'''
   Generator network. There are technically two generator networks, so this acts
   as both by using the 'direction' variable for naming.
'''
def netG(inputs, direction, reuse=False):

   padded_inputs = tf.pad(inputs, [[0,0], [3,3], [3,3], [0,0]], 'REFLECT')
   
   print 'inputs:',inputs
   print 'padded_inputs:',padded_inputs

   # c7s1-32
   conv1 = tf.layers.conv2d(padded_inputs, 32, 7, strides=1, name='g-conv1-'+direction, padding='VALID', reuse=reuse)
   conv1 = instance_norm(conv1)
   conv1 = tf.nn.relu(conv1)
   print 'conv1:',conv1

   # d64
   conv2 = tf.layers.conv2d(conv1, 64, 3, strides=2, name='g-conv2-'+direction, padding='SAME', reuse=reuse)
   conv2 = instance_norm(conv2)
   conv2 = tf.nn.relu(conv2)
   print 'conv2:',conv2
   
   # d128
   conv3 = tf.layers.conv2d(conv2, 128, 3, strides=2, name='g-conv3-'+direction, padding='SAME', reuse=reuse)
   conv3 = instance_norm(conv3)
   conv3 = tf.nn.relu(conv3)
   print 'conv3:',conv3

   # R128
   r1 = Rk(conv3,'g-'+direction+'-res-0', reuse=reuse)
   print 'r1:',r1
   r2 = Rk(r1,'g-'+direction+'-res-1', reuse=reuse)
   print 'r2:',r2
   r3 = Rk(r2,'g-'+direction+'-res-2', reuse=reuse)
   print 'r3:',r3
   r4 = Rk(r3,'g-'+direction+'-res-3', reuse=reuse)
   print 'r4:',r4
   r5 = Rk(r4,'g-'+direction+'-res-4', reuse=reuse)
   print 'r5:',r5
   r6 = Rk(r5,'g-'+direction+'-res-5', reuse=reuse)
   print 'r6:',r6
   r7 = Rk(r6,'g-'+direction+'-res-6', reuse=reuse)
   print 'r7:',r7
   r8 = Rk(r7,'g-'+direction+'-res-7', reuse=reuse)
   print 'r8:',r8
   r9 = Rk(r8,'g-'+direction+'-res-8', reuse=reuse)
   print 'r9:',r9

   t_conv1 = tf.layers.conv2d_transpose(r9, 64, 3, strides=2, name='g-t_conv1-'+direction, padding='SAME', reuse=reuse)
   t_conv1 = instance_norm(t_conv1)
   t_conv1 = tf.nn.relu(t_conv1)
   print 't_conv1:',t_conv1
   
   t_conv2 = tf.layers.conv2d_transpose(t_conv1, 3, 3, strides=2, name='g-t_conv2-'+direction, padding='SAME', reuse=reuse)
   t_conv2 = tf.nn.tanh(t_conv2)
   print 't_conv2:',t_conv2
   print
   return t_conv2

def netD(inputs, direction, reuse=False):
   print
   print 'netD'
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):

      conv1 = tf.layers.conv2d(inputs, 64, 4, strides=2, name='d-'+direction+'-conv1', padding='SAME')
      conv1 = lrelu(conv1)

      conv2 = tf.layers.conv2d(conv1, 128, 4, strides=2, name='d-'+direction+'-conv2', padding='SAME')
      conv2 = instance_norm(conv2)
      conv2 = lrelu(conv2)
      
      conv3 = tf.layers.conv2d(conv2, 256, 4, strides=2, name='d-'+direction+'-conv3', padding='SAME')
      conv3 = instance_norm(conv3)
      conv3 = lrelu(conv3)
      
      conv4 = tf.layers.conv2d(conv3, 512, 4, strides=2, name='d-'+direction+'-conv4', padding='SAME')
      conv4 = instance_norm(conv4)
      conv4 = lrelu(conv4)
      
      conv5 = tf.layers.conv2d(conv4, 1, 4, strides=2, name='d-'+direction+'-conv5', padding='SAME')
      #conv5 = instance_norm(conv5)

      print conv1
      print conv2
      print conv3
      print conv4
      print conv5
      return conv5

