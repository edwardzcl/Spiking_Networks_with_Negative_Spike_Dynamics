#-*- encoding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
#import cv2
from PIL import Image
import numpy as np
import os
from multiprocessing import Process, Queue
import sys
import time
import random
import math
 
imageWidth = 224
imageHeight = 224
imageDepth = 3
#batch_size = 128
resize_min = 256
train_dataset_path = "./train_dataset/"
valid_dataset_path = "./valid_dataset/"


def _parse_function(file_name_list, is_train=None):
    filename_queue = tf.train.string_input_producer(file_name_list)
    reader = tf.TFRecordReader()
    _, example_proto = reader.read(filename_queue)
    #dataset_train = tf.data.TFRecordDataset('train_data_3.tfrecord')
    #_img, _label = _parse_function(serialized_example)

    #filename_queue = tf.train.string_input_producer([filename])
    features = {"image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64),
                "height": tf.FixedLenFeature([], tf.int64),
                "width": tf.FixedLenFeature([], tf.int64),
               }
    parsed_features = tf.parse_single_example(example_proto, features)
 
    _height = tf.cast(parsed_features['height'], tf.int32)
    _width = tf.cast(parsed_features['width'], tf.int32)
    _label = tf.cast(parsed_features['label'], tf.int32) 
    _img = tf.decode_raw(parsed_features['image'], tf.float32)
    _img = tf.reshape(_img, [_height, _width, 3])
    #image_decoded = tf.cast(tf.image.decode_jpeg(parsed_features["image"], channels=3), tf.float32)

    #按照RESNET论文的训练图像的处理方式，对图片的短边随机缩放到256-481之间的数值，然后在随机
    #剪切224×224大小的图片。
    #random_s = tf.random_uniform([1], minval=256, maxval=481, dtype=tf.int32)[0]
    #resized_height, resized_width = tf.cond(_height<_width, 
    #            lambda: (random_s, tf.cast(tf.multiply(tf.cast(_width, tf.float64),tf.divide(random_s,_height)), tf.int32)), 
    #            lambda: (tf.cast(tf.multiply(tf.cast(_height, tf.float64),tf.divide(random_s,_width)), tf.int32), random_s))
    
    if is_train == True:  
        resized_height, resized_width = tf.cond(_height<_width,
            lambda: (resize_min, tf.cast(tf.multiply(tf.cast(_width, tf.float64),tf.divide(resize_min,_height)), tf.int32)),
            lambda: (tf.cast(tf.multiply(tf.cast(_height, tf.float64),tf.divide(resize_min,_width)), tf.int32), resize_min))

        image_float = tf.image.convert_image_dtype(_img, tf.float32)
        image_resized = tf.image.resize_images(image_float, [resized_height, resized_width])
    
        image_cropped = tf.random_crop(image_resized, [imageHeight, imageWidth, imageDepth])
        image_flipped = tf.image.random_flip_left_right(image_cropped)
        image_distorted = tf.image.random_brightness(image_flipped, max_delta=63)
        image_distorted = tf.image.random_contrast(image_distorted, lower=0.2, upper=1.8)
        #need be scaled
        #image_distorted = tf.image.random_saturation(image_distorted, lower=0.5, upper=1.5)#饱和度
        #image_distorted = tf.image.random_hue(image_distorted, max_delta=0.2)#色相
        # 5. Subtract off the mean and divide by the variance of the pixels.
        try:  # TF 0.12+
            _img = tf.image.per_image_standardization(image_distorted)
        except Exception:  # earlier TF versions
            _img = tf.image.per_image_whitening(image_distorted)


        #image_distorted = tf.image.per_image_standardization(image_distorted)
        #image_distorted = tf.transpose(image_distorted, perm=[2, 0, 1])
        #甚至另外一篇博客，别的库由其他的数据处理方式

    elif is_train == False:
        resized_height, resized_width = tf.cond(_height<_width,
            lambda: (resize_min, tf.cast(tf.multiply(tf.cast(_width, tf.float64),tf.divide(resize_min,_height)), tf.int32)),
            lambda: (tf.cast(tf.multiply(tf.cast(_height, tf.float64),tf.divide(resize_min,_width)), tf.int32), resize_min))
        image_float = tf.image.convert_image_dtype(_img, tf.float32)
        image_resized = tf.image.resize_images(image_float, [resized_height, resized_width])
    
        # calculate how many to be center crop
        shape = tf.shape(image_resized)  
        height, width = shape[0], shape[1]
        amount_to_be_cropped_h = (height - imageHeight)
        crop_top = amount_to_be_cropped_h // 2
        amount_to_be_cropped_w = (width - imageWidth)
        crop_left = amount_to_be_cropped_w // 2
        image_cropped = tf.slice(image_resized, [crop_top, crop_left, 0], [imageHeight, imageWidth, -1])
        #image_cropped = tf.image.per_image_standardization(image_cropped)
        try:  # TF 0.12+
            _img = tf.image.per_image_standardization(image_cropped)
        except Exception:  # earlier TF versions
            _img = tf.image.per_image_whitening(image_cropped)

    #if is_train == True:
    #    # 1. Randomly crop a [height, width] section of the image.
    #    img = tf.random_crop(img, [24, 24, 3])
    #    # 2. Randomly flip the image horizontally.
    #    img = tf.image.random_flip_left_right(img)
    #    # 3. Randomly change brightness.
    #    img = tf.image.random_brightness(img, max_delta=63)
    #    # 4. Randomly change contrast.
    #    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    #    # 5. Subtract off the mean and divide by the variance of the pixels.
    #    try:  # TF 0.12+
    #        img = tf.image.per_image_standardization(img)
    #    except Exception:  # earlier TF versions
    #        img = tf.image.per_image_whitening(img)

    #elif is_train == False:
    #    # 1. Crop the central [height, width] of the image.
    #    img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
    #    # 2. Subtract off the mean and divide by the variance of the pixels.
    #    try:  # TF 0.12+
    #        img = tf.image.per_image_standardization(img)
    #    except Exception:  # earlier TF versions
    #        img = tf.image.per_image_whitening(img)
    elif is_train == None:
        _img = _img

    return _img, _label
 
