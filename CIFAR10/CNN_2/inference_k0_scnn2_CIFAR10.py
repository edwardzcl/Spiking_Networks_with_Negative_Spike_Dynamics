import numpy as np
import pdb

# import spiking function
from spiking_ulils import label_encoder
from spiking_ulils import Conv2d, BatchNorm2d, Relu
from spiking_ulils import Flatten
from spiking_ulils import Linear


class MyNet():
    def __init__(self):
        self.conv0 = Conv2d(in_channels=3, n_filter=128, filter_size=(3, 3), padding=1, stride=1, use_ternary=False)   
        self.bn0 = BatchNorm2d(n_channel=128, momentum=0.1)
        self.relu0 = Relu()

        self.conv1 = Conv2d(in_channels=128, n_filter=256, filter_size=(3, 3), padding=1, stride=1, use_ternary=False)   
        self.bn1 = BatchNorm2d(n_channel=256, momentum=0.1)
        self.relu1 = Relu()

        self.conv2 = Conv2d(in_channels=256, n_filter=256, filter_size=(2, 2), padding=0, stride=2, use_ternary=False)
        self.bn2 = BatchNorm2d(n_channel=256, momentum=0.1)
        self.relu2 = Relu()

        self.conv3 = Conv2d(in_channels=256, n_filter=512, filter_size=(3, 3), padding=1, stride=1, use_ternary=False)
        self.bn3 = BatchNorm2d(n_channel=512, momentum=0.1)
        self.relu3 = Relu()

        self.conv4 = Conv2d(in_channels=512, n_filter=512, filter_size=(2, 2), padding=0, stride=2, use_ternary=False)
        self.bn4 = BatchNorm2d(n_channel=512, momentum=0.1)
        self.relu4 = Relu()

        self.conv5 = Conv2d(in_channels=512, n_filter=1024, filter_size=(3, 3), padding=1, stride=1, use_ternary=False)
        self.bn5 = BatchNorm2d(n_channel=1024, momentum=0.1)
        self.relu5 = Relu()

        self.conv6 = Conv2d(in_channels=1024, n_filter=512, filter_size=(3, 3), padding=1, stride=1, use_ternary=False)
        self.bn6 = BatchNorm2d(n_channel=512, momentum=0.1)
        self.relu6 = Relu()

        self.conv7 = Conv2d(in_channels=512, n_filter=512, filter_size=(2, 2), padding=0, stride=2, use_ternary=False)
        self.bn7 = BatchNorm2d(n_channel=512, momentum=0.1)
        self.relu7 = Relu()

        self.conv8 = Conv2d(in_channels=512, n_filter=1024, filter_size=(3, 3), padding=0, stride=1, use_ternary=False)
        self.bn8 = BatchNorm2d(n_channel=1024, momentum=0.1)
        self.relu8 = Relu()

        self.conv9 = Conv2d(in_channels=1024, n_filter=512, filter_size=(1, 1), padding=0, stride=1, use_ternary=False)
        self.bn9 = BatchNorm2d(n_channel=512, momentum=0.1)
        self.relu9 = Relu()

        self.flatten = Flatten()
        
        # ȫ���Ӳ�
        self.fc1 = Linear(dim_in=512, dim_out=10, use_ternary=False)

        
        self.parameters = self.conv0.params + self.bn0.params + self.conv1.params + self.bn1.params + self.conv2.params + self.bn2.params + \
                            self.conv3.params + self.bn3.params + self.conv4.params + self.bn4.params + self.conv5.params + self.bn5.params + self.conv6.params + self.bn6.params + \
                            self.conv7.params + self.bn7.params + self.conv8.params + self.bn8.params + self.conv9.params + self.bn9.params + \
                            self.fc1.params
        
        self.dummy_layers = [self.conv0, self.bn0, self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.conv4, self.bn4, \
        self.conv5, self.bn5, self.conv6, self.bn6, self.conv7, self.bn7, self.conv8, self.bn8, self.conv9, self.bn9, \
        self.fc1]                                        
    
    def __call__(self, X, t, mode='train'):
        """
        mode: train or test
        """
        return self.forward(X, t, mode)
    # spiking network inference during multiple time steps
    def forward(self, X, t, mode):

        # the first layer is usually a pixel-to-spike encoding layer
        conv0_out, conv0_spike_collect, conv0_spike_num, conv0_sop_num = self.conv0(X, t)

        conv1_out, conv1_spike_collect, conv1_spike_num, conv1_sop_num = self.conv1(conv0_out, t)
        
        conv2_out, conv2_spike_collect, conv2_spike_num, conv2_sop_num = self.conv2(conv1_out, t)

        conv3_out, conv3_spike_collect, conv3_spike_num, conv3_sop_num = self.conv3(conv2_out, t)

        conv4_out, conv4_spike_collect, conv4_spike_num, conv4_sop_num = self.conv4(conv3_out, t)
        
        conv5_out, conv5_spike_collect, conv5_spike_num, conv5_sop_num = self.conv5(conv4_out, t)

        conv6_out, conv6_spike_collect, conv6_spike_num, conv6_sop_num = self.conv6(conv5_out, t)

        conv7_out, conv7_spike_collect, conv7_spike_num, conv7_sop_num = self.conv7(conv6_out, t)
        
        conv8_out, conv8_spike_collect, conv8_spike_num, conv8_sop_num = self.conv8(conv7_out, t)

        conv9_out, conv9_spike_collect, conv9_spike_num, conv9_sop_num = self.conv9(conv8_out, t)
        
        flat_out = self.flatten(conv9_spike_collect, t)

        # the last layer output the membrane potential value indexing category      
        fc1_out = self.fc1(flat_out, t)

        # spike number
        spike_num = conv0_spike_num + conv1_spike_num + conv2_spike_num + conv3_spike_num + conv4_spike_num + \
        conv5_spike_num + conv6_spike_num + conv7_spike_num + conv8_spike_num + conv9_spike_num
        # spike collector
        spike_collect = np.sum(conv0_spike_collect) + np.sum(conv1_spike_collect) + np.sum(conv2_spike_collect) + np.sum(conv3_spike_collect) + np.sum(conv4_spike_collect) + \
        np.sum(conv5_spike_collect) + np.sum(conv6_spike_collect) + np.sum(conv7_spike_collect) + np.sum(conv8_spike_collect) + np.sum(conv9_spike_collect)
        # synaptic operations
        sop_num = conv1_sop_num + conv2_sop_num + conv3_sop_num + conv4_sop_num + conv5_sop_num + conv6_sop_num + conv7_sop_num + conv8_sop_num + \
        conv9_sop_num + conv9_spike_num * 10
        
        return fc1_out, spike_num, spike_collect, sop_num       
    

    def convert_assign_params(self, params, quant_level):
        tag = 0
        
        for index, layer in enumerate(self.dummy_layers):
            
            if layer.type == 'conv':         
               #self.layers[index].params[0] = params[tag].transpose(3, 2, 0, 1)
               # in this paper, we didn't quantize the weights, use_ternary is always false
               if layer.use_ternary:
                   self.dummy_layers[index].params[0][:,:,:,:] = self._ternary_operation(params[tag].transpose(3, 2, 0, 1))
               else:
                   self.dummy_layers[index].params[0][:,:,:,:] = params[tag].transpose(3, 2, 0, 1)
               tag = tag + 1
            elif layer.type == 'bn':
                # BN layers need to be scaled
                self.dummy_layers[index-1].params[3][:] = 1 / 2**quant_level * 2**quant_level * np.sqrt(params[tag+3] + 1e-5) / params[tag+1]
                self.dummy_layers[index-1].params[2][:] = (1 / 2**(quant_level+1) - params[tag]) * (2**quant_level * np.sqrt(params[tag+3] + 1e-5)) / params[tag+1] + \
                2**quant_level * params[tag+2] - self.dummy_layers[index-1].params[3][:]
                tag = tag + 4
            elif layer.type == 'fc':
                # just like the convolutional layer
                if layer.use_ternary:
                    self.dummy_layers[index].params[0][:,:] = self._ternary_operation(params[tag])
                else:
                    self.dummy_layers[index].params[0][:,:] = params[tag]
                tag = tag + 1


    def _compute_threshold(self, x):
        """
        ref: https://github.com/XJTUWYD/TWN
        Computing the threshold.
        """
        x_sum = np.sum(np.abs(x))
        threshold = x_sum / np.array([np.size(x)]).astype(np.float64)
        threshold = 0.7 * threshold
        return threshold


    def _ternary_operation(self, x):
        """
        Ternary operation use threshold computed with weights.
        """
        threshold = self._compute_threshold(x)
        x = np.sign(np.add(np.sign(np.add(x, threshold)), np.sign(np.add(x, -threshold))))
        return x



def test(sess, test_images, quant_level, test_labels, network, n_data, batch_size, time_step):
    """
    function: snn test function entrance, test_labels need use one hot encoding
    return: generate four log files: spike_num.txt, spike_collect.txt, sop_num, accuracy.txt and final SNN accuracy on test set
    """
    f1 = open('spike_num.txt', 'w')
    f2 = open('spike_collect.txt', 'w')
    f3 = open('sop_num.txt', 'w')
    f4 = open('accuracy.txt', 'w')
    n_correct = 0
    for i in range(0, n_data, batch_size):
        # generate batch datas
        batch_datas, batch_labels = sess.run([test_images, test_labels])
        batch_datas = batch_datas.transpose(0, 3, 1, 2) * 2**quant_level
        batch_labels = np.array(batch_labels, np.int32)
        batch_labels = label_encoder(batch_labels, 10)

        # time step simulation
        for t in range(time_step):
            if t == 0:
                net_out, spike_num, spike_collect, sop_num = network(batch_datas, t, mode='test')
                predict = np.argmax(net_out, axis=1)
                f1.write(str(spike_num) + '\n')
                f2.write(str(spike_collect) + '\n')
                f3.write(str(sop_num) + '\n')
                f4.write(str(np.sum(predict == np.argmax(batch_labels, axis=1))) + '\n')
            else:
                net_out, spike_num, spike_collect, sop_num = network(np.zeros_like(batch_datas), t, mode='test')
                predict = np.argmax(net_out, axis=1)
                f1.write(str(spike_num) + '\n')
                f2.write(str(spike_collect) + '\n')
                f3.write(str(sop_num) + '\n')
                f4.write(str(np.sum(predict == np.argmax(batch_labels, axis=1))) + '\n')
        n_correct += np.sum(predict == np.argmax(batch_labels, axis=1))
        print('-----------------------Batch_number: ', i / batch_size, ' completed-----------------------')
        print(np.sum(predict == np.argmax(batch_labels, axis=1)) / batch_size)
        
    test_acc = n_correct / n_data
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    return test_acc


import os
import time

import tensorlayer as tl
import tensorflow as tf

tf.reset_default_graph()

## Download data, and convert to TFRecord format, see ```tutorial_tfrecord.py```
X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

print('X_train.shape', X_train.shape)  # (50000, 32, 32, 3)
print('y_train.shape', y_train.shape)  # (50000,)
print('X_test.shape', X_test.shape)  # (10000, 32, 32, 3)
print('y_test.shape', y_test.shape)  # (10000,)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))

def data_to_tfrecord(images, labels, filename):
    """ Save data into TFRecord """
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    print("Converting data into %s ..." % filename)
    # cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        ## Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        # print(label)
        ## Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                }
            )
        )
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    if is_train ==True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])
        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)
        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)
        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        try:  # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except Exception:  # earlier TF versions
            img = tf.image.per_image_whitening(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        try:  # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except Exception:  # earlier TF versions
            img = tf.image.per_image_whitening(img)
    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label


## Save data into TFRecord files
data_to_tfrecord(images=X_train, labels=y_train, filename="train.cifar10")
data_to_tfrecord(images=X_test, labels=y_test, filename="test.cifar10")

batch_size = 50

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # prepare data in cpu
    x_train_, y_train_ = read_and_decode("train.cifar10", True)
    x_test_, y_test_ = read_and_decode("test.cifar10", False)
    # set the number of threads here
    x_train_batch, y_train_batch = tf.train.shuffle_batch(
        [x_train_, y_train_], batch_size=batch_size, capacity=2000, min_after_dequeue=1000, num_threads=32
    )
    # for testing, uses batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch(
        [x_test_, y_test_], batch_size=batch_size, capacity=50000, num_threads=32
    )

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # define SNN instance
    mynet = MyNet()

    # load parameter
    model = np.load('model_cifar_10.npz')
    params = model['params']

    # quantization level k
    quant_level = 0
    mynet.convert_assign_params(params, quant_level)

    # total time steps
    time_step = 100

    test_acc = test(sess, x_test_batch, quant_level, y_test_batch, network=mynet, n_data=y_test.shape[0], batch_size=batch_size, time_step=time_step)

    print(test_acc)

    coord.request_stop()
    coord.join(threads)
    sess.close()

