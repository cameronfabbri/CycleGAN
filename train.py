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

from tf_ops import *
import data_ops

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--ARCHITECTURE', required=True,help='Architecture for the generator')
   parser.add_argument('--EPOCHS',       required=False,default=100,type=int,help='Number of epochs for GAN')
   parser.add_argument('--DATASET',      required=True,help='The dataset to use')
   parser.add_argument('--LEARNING_RATE',required=False,default=0.0002,type=float,help='Learning rate for the pretrained network')
   parser.add_argument('--BATCH_SIZE',   required=False,type=int,default=32,help='Batch size to use')
   parser.add_argument('--LOSS_METHOD',  required=False,default='wasserstein',help='Loss function for GAN',
      choices=['wasserstein','least_squares','energy','gan','cnn'])
   parser.add_argument('--CYCLE_WEIGHT', required=False,help='weight of L2 for combined loss',type=float,default=0.0)
   a = parser.parse_args()

   ARCHITECTURE  = a.ARCHITECTURE
   EPOCHS        = a.EPOCHS
   DATASET       = a.DATASET
   LEARNING_RATE = a.LEARNING_RATE
   BATCH_SIZE    = a.BATCH_SIZE
   LOSS_METHOD   = a.LOSS_METHOD
   CYCLE_WEIGHT  = a.CYCLE_WEIGHT
   
   EXPERIMENT_DIR = 'checkpoints/ARCHITECTURE_'+ARCHITECTURE+'/DATASET_'+DATASET+'/LEARNING_RATE_'+str(LEARNING_RATE)+'/LOSS_'+LOSS_METHOD+'/CYCLE_'+str(CYCLE_WEIGHT)+'/'
   print EXPERIMENT_DIR
   
   IMAGES_DIR = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.makedirs('checkpoints/')
   except: pass
   try: os.makedirs(EXPERIMENT_DIR)
   except: pass
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
   
   # The image from data domain A
   inputsA = Data.inputsA
   
   # The image from data domain B
   inputsB = Data.inputsB

   # TODO it's actually two different generators, not just one
   # transform domain A to domain B, then back to domain A, minimize L1
   AtoB = network.netG(inputsA)
   BtoA = network.netG(inputsB)

   Arecon = network.netG(BtoA)
   Brecon = network.netG(AtoB)

   AR_loss = tf.reduce_mean(tf.abs(Arecon-inputsA))
   BR_loss = tf.reduce_mean(tf.abs(Brecon-inputsB))

   L_cyc = tf.reduce_mean(tf.abs(AR_loss-BR_loss))

   e = 1e-12
   if LOSS_METHOD == 'least_squares':
      print 'Using least squares loss'
      # Least squares requires sigmoid activation on D
      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      
      gen_loss_GAN = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1  = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2  = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      elif L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using GAN loss, no L1 or L2'
         errG = gen_loss_GAN
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
   if LOSS_METHOD == 'gan':
      print 'Using original GAN loss'
      if LOSS_METHOD is not 'cnn':
         D_real = tf.nn.sigmoid(D_real)
         D_fake = tf.nn.sigmoid(D_fake)
         gen_loss_GAN = tf.reduce_mean(-tf.log(D_fake + e))
      else: gen_loss_GAN = 0.0
      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1  = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2  = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      if L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using GAN loss, no L1 or L2'
         #errD = errD + e
         errG = gen_loss_GAN
      errD = tf.reduce_mean(-(tf.log(D_real+e)+tf.log(1-D_fake+e)))
   
   if LOSS_METHOD == 'energy':
      print 'Using energy loss'
      margin = 80
      gen_loss_GAN = D_fake

      zero = tf.zeros_like(margin-D_fake)

      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1 = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG        = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2 = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG        = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      if L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using energy loss, no L1 or L2'
         errG = gen_loss_GAN
      errD = D_real + tf.maximum(zero, margin-D_fake)

   # tensorboard summaries
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   if LOSS_METHOD == 'wasserstein':
      # clip weights in D
      clip_values = [-0.005, 0.005]
      clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for var in d_vars]

   # MSE loss for pretraining
   if EPOCHS > 0:
      print 'Pretraining generator for',EPOCHS,'epochs...'
      mse_loss = tf.reduce_mean((ab_image-gen_ab)**2)
      #mse_train_op = tf.train.AdamOptimizer(learning_rate=PRETRAIN_LR,beta1=0.5).minimize(mse_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      mse_train_op = tf.train.AdamOptimizer(learning_rate=PRETRAIN_LR).minimize(mse_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      tf.add_to_collection('vars', mse_train_op)
      tf.summary.scalar('mse_loss', mse_loss)
   if LOSS_METHOD == 'wasserstein':
      G_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=0.9).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      D_train_op = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=0.9).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)
   else:
      #G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=0.5).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      #D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=0.5).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)
      G_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      D_train_op = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   # only keep one model
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
   if LOAD_MODEL:
      ckpt = tf.train.get_checkpoint_state(LOAD_MODEL)
      print "Restoring model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         raise
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
         sess.run(mse_train_op)
         mse, summary = sess.run([mse_loss, merged_summary_op])
         step += 1
         summary_writer.add_summary(summary, step)
         print 'step:',step,'mse:',mse,'time:',time.time()-s
         if step % 500 == 0:
            print
            print 'Saving model'
            print
            saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
            saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
      if EPOCHS > 0:
         print 'Done pretraining....training D and G now'
         epoch_num = 0
      while epoch_num < EPOCHS:
         epoch_num = step/(num_train/BATCH_SIZE)
         s = time.time()

         if LOSS_METHOD == 'wasserstein':
            if step < 10 or step % 500 == 0:
               n_critic = 100
            else: n_critic = NUM_CRITIC
            for critic_itr in range(n_critic):
               sess.run(D_train_op)
               sess.run(clip_discriminator_var_op)
            sess.run(G_train_op)
            D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])
         
         elif LOSS_METHOD == 'least_squares':
            sess.run(D_train_op)
            sess.run(G_train_op)
            D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])
         elif LOSS_METHOD == 'gan'or LOSS_METHOD == 'energy':
            sess.run(D_train_op)
            sess.run(G_train_op)
            D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])
         elif LOSS_METHOD == 'cnn':
            sess.run(G_train_op)
            loss, summary = sess.run([errG, merged_summary_op])

         summary_writer.add_summary(summary, step)
         if LOSS_METHOD != 'cnn' and step%10==0: print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-s
         else:
            if step%50==0:print 'epoch:',epoch_num,'step:',step,'loss:',loss,' time:',time.time()-s
         step += 1
         
         if step%500 == 0:
            print 'Saving model...'
            saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
            saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
            print 'Model saved\n'

      print 'Finished training', time.time()-start
      saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
      saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
      exit()
