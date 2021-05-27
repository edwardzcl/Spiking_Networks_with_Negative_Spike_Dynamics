# -*- coding: utf-8 -*-
"""
Created on Fri May  4 18:25:59 2018

@author: ***
Evaluation script for full-precision MLP on N-MNIST.
"""

#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorlayer as tl 
import os
print(os.getcwd())

tf.reset_default_graph()

X_train, y_train, X_test, y_test = tl.files.load_n_mnist_dataset(shape=(-1, 34, 34, 1))
# X_train, y_train, X_test, y_test = tl.files.load_cropped_svhn(include_extra=False)

sess = tf.InteractiveSession()

batch_size = 200

x = tf.placeholder(tf.float32, shape=[None, 34, 34, 1])
y_ = tf.placeholder(tf.int64, shape=[None])

model_file_name = "./model_n_mnist_advanced.ckpt"
resume = True # load model, resume from previous checkpoint?

def model(x, is_train=True, reuse=False):
    # In BNN, all the layers inputs are binary, with the exception of the first layer.
    # ref: https://github.com/itayhubara/BinaryNet.tf/blob/master/models/BNN_cifar10.py
    with tf.variable_scope("binarynet", reuse=reuse):
        net = tl.layers.InputLayer(x, name='input')
        #net = tl.layers.Conv2d(net, n_filter=12, filter_size=(5, 5), strides=(1, 1), padding='SAME', b_init=None, name='bcnn0')
        #net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn0')
        #net = tl.layers.Quant_Layer(net, k)
        #net = tl.layers.SignLayer(net)

        net = tl.layers.Conv2d(net, n_filter=400, filter_size=(34, 34), strides=(1, 1), padding='VALID', b_init=None, name='bcnn1')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn1')
        #net = tl.layers.Quant_Layer(net, k)
        #net = tl.layers.SignLayer(net)
        #net.outputs = (net.outputs+1)/2

        net1 = tl.layers.Conv2d(net, n_filter=400, filter_size=(1, 1), strides=(1, 1), padding='VALID', b_init=None, name='bcnn2')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
        net = tl.layers.BatchNormLayer(net1, act=tf.nn.relu, is_train=is_train, name='bn2')
        #net = tl.layers.Quant_Layer(net, k)
        #net = tl.layers.SignLayer(net)
        #net.outputs = (net.outputs+1)/2

        net = tl.layers.FlattenLayer(net)
        # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop1')
        #net = tl.layers.DenseLayer(net, n_units=256, b_init=None, name='dense')
        #net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn5')
        #net = tl.layers.SignLayer(net)


        # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop2')
        #net = tl.layers.SignLayer(net)
        #net.outputs = (net.outputs+1)/2

        net = tl.layers.DenseLayer(net, 10, b_init=None, name='bout')
        #net = tl.layers.BatchNormLayer(net, is_train=is_train, name='bno')
    return net, net1


# define inferences
net_train, _ = model(x, is_train=True, reuse=False)
net_test,net1 = model(x, is_train=False, reuse=True)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

if resume:
    print("Load existing model " + "!" * 10)
    saver = tf.train.Saver()
    saver.restore(sess, model_file_name)

net_train.print_params()
net_train.print_layers()

# print(sess.run(net_test.all_params)) # print real values of parameters

# test
print('Evaluation')
test_loss, test_acc, n_batch = 0, 0, 0
#for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
err, ac = sess.run([cost_test, acc], feed_dict={x: X_test[:,:,:,:], y_: y_test[:]})
test_loss += err
test_acc += ac
#    n_batch += 1
x = sess.run(net1.outputs[0,:,0,0],feed_dict={x: X_test[:,:,:,:], y_: y_test[:]})
#    break
print("   test loss: %f" % (test_loss / 1))
print("   test acc: %f" % (test_acc / 1))
print(x)



"""

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


x=x.flatten()
mu =np.mean(x) #计算均值
sigma =np.std(x)

mu =np.mean(x) #计算均值
sigma =np.std(x)

print(mu,sigma)
num_bins = 60 #直方图柱子的数量

n, bins, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.8)
#直方图函数，x为x轴的值，normed=1表示为概率密度，即和为一，绿色方块，色深参数0.5.返回n个概率，直方块左边线的x值，及各个方块对象
y = mlab.normpdf(bins, mu, sigma)#拟合一条最佳正态分布曲线y 
plt.plot(bins, y, 'r--') #绘制y的曲线
#plt.xlabel('s') #绘制x轴
#plt.ylabel('BN(s)') #绘制y轴
#plt.title(r'$\mu=-0.04657$, $\sigma=1.01954$')#中文标题 u'xxx' 

plt.subplots_adjust(left=0.15)#左边距 
plt.show() 

"""
