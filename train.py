import cPickle as pickle
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'nets/')

from tf_ops import *
import data_ops

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--ARCHITECTURE', required=False,default='resnet',help='Architecture for the generator')
   parser.add_argument('--EPOCHS',       required=False,default=100,type=int,help='Number of epochs for GAN')
   parser.add_argument('--DATASET',      required=True,help='The dataset to use')
   parser.add_argument('--LEARNING_RATE',required=False,default=0.0002,type=float,help='Learning rate for the pretrained network')
   parser.add_argument('--BATCH_SIZE',   required=False,type=int,default=4,help='Batch size to use')
   parser.add_argument('--LOSS_METHOD',  required=False,default='least_squares',help='Loss function for GAN',
      choices=['wasserstein','least_squares','gan'])
   parser.add_argument('--CYCLE_WEIGHT', required=False,help='weight of the cycle loss',type=float,default=10.0)
   parser.add_argument('--HISTORY',      required=False,help='size of history',type=int,default=50)
   a = parser.parse_args()

   ARCHITECTURE  = a.ARCHITECTURE
   EPOCHS        = a.EPOCHS
   DATASET       = a.DATASET
   LEARNING_RATE = a.LEARNING_RATE
   BATCH_SIZE    = a.BATCH_SIZE
   LOSS_METHOD   = a.LOSS_METHOD
   CYCLE_WEIGHT  = a.CYCLE_WEIGHT

   import resnet as network

   EXPERIMENT_DIR = 'checkpoints/ARCHITECTURE_'+ARCHITECTURE+'/DATASET_'+DATASET+'/LEARNING_RATE_'+str(LEARNING_RATE)+'/LOSS_'+LOSS_METHOD+'/CYCLE_'+str(CYCLE_WEIGHT)+'/'
   print EXPERIMENT_DIR
   
   IMAGES_DIR = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.makedirs(IMAGES_DIR)
   except: pass
   
   # write all this info to a pickle file in the experiments directory
   exp_info = dict()
   exp_info['ARCHITECTURE']  = ARCHITECTURE
   exp_info['EPOCHS']        = EPOCHS
   exp_info['DATASET']       = DATASET
   exp_info['LEARNING_RATE'] = LEARNING_RATE
   exp_info['BATCH_SIZE']    = BATCH_SIZE
   exp_info['LOSS_METHOD']   = LOSS_METHOD
   exp_info['CYCLE_WEIGHT']  = CYCLE_WEIGHT
   exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'ARCHITECTURE:  ',ARCHITECTURE
   print 'EPOCHS:        ',EPOCHS
   print 'DATASET:       ',DATASET
   print 'LEARNING_RATE: ',LEARNING_RATE
   print 'BATCH_SIZE:    ',BATCH_SIZE
   print 'LOSS_METHOD:   ',LOSS_METHOD
   print 'CYCLE_WEIGHT:  ',CYCLE_WEIGHT
   print

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # load data from ops/data_ops.py
   Data = data_ops.loadData(DATASET, BATCH_SIZE)

   # number of training images
   num_train = Data.count
   
   # The image from data domain A in range [-1,1]
   inputsA = data_ops.preprocess(Data.inputsA)
   
   # The image from data domain B in range [-1,1]
   inputsB = data_ops.preprocess(Data.inputsB)

   # generate images
   fakeB = network.netG(inputsA, 'atob') # A to B
   fakeA = network.netG(inputsB, 'btoa') # B to A

   # cycle part for reconstruction
   Brecon = network.netG(fakeA, 'atob', reuse=True) # A to B
   Arecon = network.netG(fakeB, 'btoa', reuse=True) # B to A

   # L1 loss for cycles
   AR_loss = CYCLE_WEIGHT*tf.reduce_mean(tf.abs(Arecon-inputsA))
   BR_loss = CYCLE_WEIGHT*tf.reduce_mean(tf.abs(Brecon-inputsB))

   # total cycle loss
   L_cyc = tf.reduce_mean(AR_loss+BR_loss)

   # send the real domain A images to d
   discA = network.netD(inputsA, 'domain_a')

   # send the real domain images B to d
   discB = network.netD(inputsA, 'domain_b')

   # send the generated domain A images to d
   discA_gen = network.netD(fakeA, 'domain_a', reuse=True)
   
   # send the generated domain B images to d
   discB_gen = network.netD(fakeA, 'domain_b', reuse=True)

   D_real = discA+discB
   D_fake = discA_gen+discB_gen
  
   # now get all testing stuff
   testData = data_ops.loadData(DATASET, BATCH_SIZE, train=False)
   num_test = testData.count

   # convert images to [-1, 1]
   test_inputsA = data_ops.preprocess(testData.inputsA)
   test_inputsB = data_ops.preprocess(testData.inputsB)
   
   # send each through the generator
   test_BtoA = data_ops.deprocess(network.netG(test_inputsB, 'atob', reuse=True))
   test_AtoB = data_ops.deprocess(network.netG(test_inputsA, 'btoa', reuse=True))

   # convert back to [0, 255]
   test_inputsA = data_ops.deprocess(test_inputsA)
   test_inputsB = data_ops.deprocess(test_inputsB)

   # added for safe log
   e = 1e-12
   if LOSS_METHOD == 'least_squares':
      print 'Using least squares loss'

      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      
      errG = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))

   if LOSS_METHOD == 'gan':
      print 'Using original GAN loss'
      D_real = tf.nn.sigmoid(D_real)
      D_fake = tf.nn.sigmoid(D_fake)
      errG = tf.reduce_mean(-tf.log(D_fake + e))

      errD = tf.reduce_mean(-(tf.log(D_real+e)+tf.log(1-D_fake+e)))

   errG = errG + L_cyc

   # tensorboard summaries
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd-' in var.name]
   g_vars = [var for var in t_vars if 'g-' in var.name]

   if LOSS_METHOD == 'wasserstein':
      # clip weights in D
      clip_values = [-0.005, 0.005]
      clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for var in d_vars]
      G_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,beta1=0.5).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      D_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE,beta1=0.5).minimize(errD, var_list=d_vars, global_step=global_step, colocate_gradients_with_ops=True)
   else:
      G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=0.5).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=0.5).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   # restore previous model if there is one
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion
   step = sess.run(global_step)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)
   merged_summary_op = tf.summary.merge_all()
   start = time.time()
   while True:
      epoch_num = step/(num_train/BATCH_SIZE)
      while epoch_num < EPOCHS:
         epoch_num = step/(num_train/BATCH_SIZE)
         s = time.time()

         if LOSS_METHOD == 'wasserstein':
            if step < 10 or step % 500 == 0:
               n_critic = 100
            else: n_critic = NUM_CRITIC
            for critic_itr in range(5):
               sess.run(D_train_op)
               sess.run(clip_discriminator_var_op)
            sess.run(G_train_op)
            D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])
         
         elif LOSS_METHOD == 'least_squares':
            sess.run(D_train_op)
            sess.run(G_train_op)
            D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])

         summary_writer.add_summary(summary, step)
         print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss
         step += 1
         
         if step%100 == 0:
            print 'Saving model...'
            saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
            saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
            print 'Model saved\n'
            
            # now get some test images (just the first in the batch)
            #real_A = sess.run(test_inputsA)[0]
            #real_B = sess.run(test_inputsB)[0]

            gen_A  = sess.run(test_BtoA)[0]
            gen_B  = sess.run(test_AtoB)[0]

            #misc.imsave(IMAGES_DIR+'real_A_'+str(step)+'.png', real_A)
            #misc.imsave(IMAGES_DIR+'real_B_'+str(step)+'.png', real_B)
            misc.imsave(IMAGES_DIR+'gen_A_'+str(step)+'.png', gen_A)
            misc.imsave(IMAGES_DIR+'gen_B_'+str(step)+'.png', gen_B)

      print 'Finished training', time.time()-start
      saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
      saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
      exit()
