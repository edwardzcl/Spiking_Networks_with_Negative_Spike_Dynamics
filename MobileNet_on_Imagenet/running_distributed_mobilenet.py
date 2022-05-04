# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:24:54 2018

@author: ZCL
"""

#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
- 1. This model has 1,068,298 paramters and Dorefa compression strategy(weight:1 bit, active: 1 bit),
after 500 epoches' training with GPU,accurcy of 41.1% was found.
- 2. For simplified CNN layers see "Convolutional layer (Simplified)"
in read the docs website.
- 3. Data augmentation without TFRecord see `tutorial_image_preprocess.py` !!
Links
-------
.. https://www.tensorflow.org/versions/r0.9/tutorials/deep_cnn/index.html
.. https://github.com/tensorflow/tensorflow/tree/r0.9/tensorflow/models/image/cifar10
Note
------
The optimizers between official code and this code are different.
Description
-----------
The images are processed as follows:
.. They are cropped to 24 x 24 pixels, centrally for evaluation or randomly for training.
.. They are approximately whitened to make the model insensitive to dynamic range.
For training, we additionally apply a series of random distortions to
artificially increase the data set size:
.. Randomly flip the image from left to right.
.. Randomly distort the image brightness.
.. Randomly distort the image contrast.
Speed Up
--------
Reading images from disk and distorting them can use a non-trivial amount
of processing time. To prevent these operations from slowing down training,
we run them inside 16 separate threads which continuously fill a TensorFlow queue.
"""

import os
import time
import tensorflow as tf
import tensorlayer as tl
import ast
import argparse

#import tfrecord_imagenet
from input_data import _parse_function
#import input_data

parser = argparse.ArgumentParser()
# resume from previous checkpoint after 100 epochs and decay learning rate by 10 times
parser.add_argument('--resume', type=ast.literal_eval, default=False)
# learning rate, # decay it by 10×  after 100 epochs, and set resume = True
parser.add_argument('--learning_rate', type=float, default=0.01)
# quantization level k, in paper, we set it to {0,1,2}, default is 1
parser.add_argument('--quant_level', type=int, default=1)
# training or inference
parser.add_argument('--mode', type=str, default='training')
args = parser.parse_args()

print(args.resume, args.learning_rate, args.quant_level, args.mode)

tf.reset_default_graph()

NUM_GPUS = 5
imageWidth = 224
imageHeight = 224
imageDepth = 3
batch_per_gpu = 200
batch_size = batch_per_gpu*NUM_GPUS
resize_min = 256
train_dataset_path = "./train_dataset/"
valid_dataset_path = "./valid_dataset/"

model_file_name = "./model_imagenet_mobilenetv1.ckpt"
#resume = False  # load model, resume from previous checkpoint?


print('X_train.shape', (1281120, 224, 224, 3))  # (50000, 32, 32, 3)
print('y_train.shape', (1281120,))  # (50000,)
#1281160
print('X_test.shape', (50000, 224, 224, 3))  # (10000, 32, 32, 3)
print('y_test.shape', (50000,))  # (10000,)
print('X %s   y %s' % ('float32', 'int64'))


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads



with tf.Graph().as_default(), tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # prepare data in cpu
    train_file_name_list = os.listdir(train_dataset_path)
    train_file_name_list = [train_dataset_path + i for i in train_file_name_list]
    valid_file_name_list = os.listdir(valid_dataset_path)
    valid_file_name_list = [valid_dataset_path + i for i in valid_file_name_list]
    x_train_, y_train_ = _parse_function(train_file_name_list, True)
    x_valid_, y_valid_ = _parse_function(valid_file_name_list, False)

    # set the number of threads here
    x_train_batch, y_train_batch = tf.train.shuffle_batch(
        [x_train_, y_train_], batch_size=batch_size, capacity=5000, min_after_dequeue=batch_size, num_threads=128
    )
    x_train_batch = tf.split(x_train_batch, NUM_GPUS)
    y_train_batch = tf.split(y_train_batch, NUM_GPUS)
    # for testing, uses batch instead of shuffle_batch
    x_valid_batch, y_valid_batch = tf.train.batch(
        [x_valid_, y_valid_], batch_size=batch_size, capacity=50000, num_threads=128
    )
    x_valid_batch = tf.split(x_valid_batch, NUM_GPUS)
    y_valid_batch = tf.split(y_valid_batch, NUM_GPUS)

    #采用管道设计
    #x = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3])
    #y_ = tf.placeholder(tf.int32, shape=[batch_size])


    def model(x_crop, y_, is_train, reuse, k=args.quant_level):
        """ For more simplified CNN APIs, check tensorlayer.org """
        with tf.variable_scope("model", reuse=reuse):
            # get model without the last layer
            net = tl.models.MobileNetV1(x_crop, k=k, end_with='out', is_train=is_train, reuse=reuse)
            # 支持量化
            # net = tl.layers.TernaryLayer(net, name='ternary21')
            # the last layer is usually high-precison refer to the previous works.

            y = net.outputs

            #Get the inference logits by the model for the training or validation images
            ##分布式有待研究

            ce = tl.cost.cross_entropy(y, y_, name='cost')
            # L2 for the MLP, without this, the accuracy will be reduced by 15%.
            L2 = 0
            for p in tl.layers.get_variables_with_name('first/', True, True):
                L2 += tf.contrib.layers.l2_regularizer(0.00004)(p)       
            for q in tl.layers.get_variables_with_name('second/W', True, True):     
                L2 += tf.contrib.layers.l2_regularizer(0.00001)(q)   
            cost = ce + L2

            # correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y), 1), y_)
            correct_prediction_top1 = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
            acc_top1 = tf.reduce_mean(tf.cast(correct_prediction_top1, tf.float32))

            correct_prediction_top5 = tf.nn.in_top_k(y, y_, k=5)
            acc_top5 = tf.reduce_mean(tf.cast(correct_prediction_top5, tf.float32))
            return net, cost, acc_top1, acc_top5

    ## You can also use placeholder to feed_dict in data after using
    ## val, l = sess.run([x_train_batch, y_train_batch]) to get the data
    # x_crop = tf.placeholder(tf.float32, shape=[batch_size, 24, 24, 3])
    # y_ = tf.placeholder(tf.int32, shape=[batch_size,])
    # cost, acc, network = model(x_crop, y_, None)
    #with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
    train_op = tf.train.AdamOptimizer(args.learning_rate)
    #后期可以把学习率衰减机制加上
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
        # gpu
        for i in range(NUM_GPUS):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('gpu_%d' % i) as scope:
                    with tf.name_scope("tower_%d" % i):
                        	#每个gpu里放不同的数据
                            #_x = x[i * batch_size:(i + 1) * batch_size]
                            #_y = y_[i * batch_size:(i + 1) * batch_size]

                            # calculate inference
                            #network, cost, acc_top1, acc_top5, = model(x[i*batch_per_gpu:(i+1)*batch_per_gpu,:,:,:], y_[i*batch_per_gpu:(i+1)*batch_per_gpu], True, False)
                            #_, cost_test, acc_test_top1, acc_test_top5 = model(x[i*batch_per_gpu:(i+1)*batch_per_gpu,:,:,:], y_[i*batch_per_gpu:(i+1)*batch_per_gpu], False, True)
                            network, cost, acc_top1, acc_top5, = model(x_train_batch[i], y_train_batch[i], True, False, k=args.quant_level)
                            _, cost_test, acc_test_top1, acc_test_top5 = model(x_valid_batch[i], y_valid_batch[i], False, True, k=args.quant_level)
                            #y = inference(_x, reuse=False, drop_rate=drop_rate)
                            # loss
                            #mse_loss = tf.losses.mean_squared_error(_y, y)
                            #cur_loss = mse_loss
							#变量共享
                            tf.get_variable_scope().reuse_variables()
                            # 当前梯度
                            cur_grad = train_op.compute_gradients(cost)
                            tower_grads.append(cur_grad)

    #后续可以把各种日志文件加上
    grads = average_gradients(tower_grads)   
    train_op = train_op.apply_gradients(grads)  
    #train_op = apply_gradient_op           


    #with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
    #    network, cost, acc_top1, acc_top5, = model(x_train_batch, y_train_batch, True, False)
    #    _, cost_test, acc_test_top1, acc_test_top5 = model(x_valid_batch, y_valid_batch, False, True)



    ## train
    n_epoch = 100
    #learning_rate = 0.01
    print_freq = 1
    n_step_epoch = int(1281120 / batch_size)
    #n_step_epoch = int(1281160 / batch_size)
    n_step = n_epoch * n_step_epoch

    #with tf.device('/gpu:0'):  # <-- remove it if you don't have GPU
    #    train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    tl.layers.initialize_global_variables(sess)
    if args.resume:
        print("Load existing model " + "!" * 10)
        saver = tf.train.Saver()
        saver.restore(sess, model_file_name)

    network.print_params(False)
    network.print_layers()

    print('   learning_rate: %f' % args.learning_rate)
    print('   batch_size: %d' % batch_size)
    print('   n_epoch: %d, step in an epoch: %d, total n_step: %d' % (n_epoch, n_step_epoch, n_step))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    if args.mode == 'training':
       for epoch in range(n_epoch):
           start_time = time.time()
           train_loss, train_acc_top1, train_acc_top5, n_batch = 0, 0, 0, 0
           for s in range(n_step_epoch):
               ## You can also use placeholder to feed_dict in data after using
               # val, l = sess.run([x_train_batch, y_train_batch])
               # tl.visualize.images2d(val, second=3, saveable=False, name='batch', dtype=np.uint8, fig_idx=2020121)
               # err, ac, _ = sess.run([cost, acc, train_op], feed_dict={x_crop: val, y_: l})
               #imgs_train, labels_train = sess.run([x_train_batch, y_train_batch])
               #err, ac_top1, ac_top5, _ = sess.run([cost, acc_top1, acc_top5, train_op], feed_dict={x: imgs_train, y_: labels_train})
               err, ac_top1, ac_top5, _ = sess.run([cost, acc_top1, acc_top5, train_op])
               print("batch_num: ", s, "batch_loss: ", err, "batch_top1_acc: ", ac_top1, "batch_top5_acc: ", ac_top5)
               step += 1
               train_loss += err
               train_acc_top1 += ac_top1
               train_acc_top5 += ac_top5
               n_batch += 1

           if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
               print(
                   "Epoch %d : Step %d-%d of %d took %fs" %
                   (epoch, step, step + n_step_epoch, n_step, time.time() - start_time)
               )
               print("   train loss: %f" % (train_loss / n_batch))
               print("   train acc_top1: %f" % (train_acc_top1 / n_batch))
               print("   train acc_top5: %f" % (train_acc_top5 / n_batch))

               test_loss, test_acc_top1, test_acc_top5, n_batch = 0, 0, 0, 0
               for _ in range(int(50000/ batch_size)):
                   #imgs_valid, labels_valid = sess.run([x_valid_batch, y_valid_batch])
                   #err, ac_top1, ac_top5 = sess.run([cost_test, acc_test_top1, acc_test_top5], feed_dict={x: imgs_valid, y_: labels_valid})
                   err, ac_top1, ac_top5 = sess.run([cost_test, acc_test_top1, acc_test_top5])
                   test_loss += err
                   test_acc_top1 += ac_top1
                   test_acc_top5 += ac_top5
                   n_batch += 1
               print("   test loss: %f" % (test_loss / n_batch))
               print("   test acc_top1: %f" % (test_acc_top1 / n_batch))
               print("   test acc_top5: %f" % (test_acc_top5 / n_batch))

           if (epoch + 1) % (print_freq * 10) == 0:
               print("Save model " + "!" * 10)
               saver = tf.train.Saver()
               save_path = saver.save(sess, model_file_name)
               # you can also save model into npz
               tl.files.save_npz(network.all_params, name='model.npz', sess=sess)
               # and restore it as follow:
               # tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=network)

    else:
        test_loss, test_acc_top1, test_acc_top5, n_batch = 0, 0, 0, 0
        for _ in range(int(50000 / batch_size)):
            #imgs_valid, labels_valid = sess.run([x_valid_batch, y_valid_batch])
            #err, ac_top1, ac_top5 = sess.run([cost_test, acc_test_top1, acc_test_top5], feed_dict={x: imgs_valid, y_: labels_valid})
            err, ac_top1, ac_top5 = sess.run([cost_test, acc_test_top1, acc_test_top5])
            test_loss += err
            test_acc_top1 += ac_top1
            test_acc_top5 += ac_top5
            n_batch += 1
        print("   test loss: %f" % (test_loss / n_batch))
        print("   test acc_top1: %f" % (test_acc_top1 / n_batch))
        print("   test acc_top5: %f" % (test_acc_top5 / n_batch))

        coord.request_stop()
        coord.join(threads)
        sess.close()


