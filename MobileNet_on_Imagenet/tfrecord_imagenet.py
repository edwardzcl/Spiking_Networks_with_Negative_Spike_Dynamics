#-*- encoding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os
from multiprocessing import Process, Queue
import sys
import time
import random
import math
 
max_num = 100000  #max record number in one file
train_path = './train/'  #the folder stroes the train images
valid_path = './val/'  #the folder stroes the validation images
cores = 8   #number of CPU cores to process
 
#Imagenet图片都保存在/data目录下，里面有1000个子目录，获取这些子目录的名字
classes = os.listdir(train_path)
 
#构建一个字典，Key是目录名，value是类名0-999
labels_dict = {}
for i in range(len(classes)):
    labels_dict[classes[i]]=i
 
#构建训练集文件列表，里面的每个元素是路径名+图片文件名+类名
train_images_name_list = []
for i in range(len(classes)):
    path = train_path+classes[i]+'/'
    images_files = os.listdir(path)
    for image_file in images_files:
        train_images_name_list.append(path+image_file)
random.shuffle(train_images_name_list)

print('number of train_images: ', len(train_images_name_list))
#1281167

 
#读取验证集的图片对应的类名标签文件
#valid_classes = []
#with open('imagenet_2012_validation_synset_labels.txt', 'r') as f:
#	  valid_classes = [line.strip() for line in f.readlines()]
#valid_classes = [i for i in range(1000)]
#构建验证集文件列表，里面的每个元素是路径名+图片文件名+类名
valid_images_name_list = []
valid_classes = os.listdir(valid_path)
for i, file_item in enumerate(valid_classes):
    #number = int(file_item[15:23])-1
    path = valid_path+valid_classes[i]+'/'
    images_files = os.listdir(path)
    for image_file in images_files:
        valid_images_name_list.append(path+image_file)
    #valid_images_labels_list.append(valid_path+','+file_item+','+valid_classes[number])

print('number of valid_images: ', len(valid_images_name_list))
#50000

#把图像数据和标签转换为TRRECORD的格式
def make_example(image, label, height, width):
    colorspace = b'RGB'
    channels = 3
    img_format = b'JPEG'
    return tf.train.Example(features=tf.train.Features(feature={
        'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label' : tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'height' : tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width' : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    }))
 
 
 
#这个函数用来生成TFRECORD文件，第一个参数是列表，每个元素是图片文件名加类名，第二个参数是写入的目录名
#第三个参数是文件名的起始序号，第四个参数是队列名称，用于和父进程发送消息
def gen_tfrecord(trainrecords, targetfolder, startnum, queue):
    tfrecords_file_num = startnum
    file_num = 0
    total_num = len(trainrecords)
    pid = os.getpid()
    queue.put((pid, file_num))
    writer = tf.python_io.TFRecordWriter(targetfolder+"data_"+str(tfrecords_file_num)+".tfrecord")
    for record in trainrecords:
        print(record)
        file_num += 1
        fields = record.split('/')
        img = cv2.imread(record)
        height, width, _ = img.shape
        height = int(height)
        width = int(width)
        img_raw = np.asarray(img, dtype=np.float32).tobytes()
        # img_jpg = cv2.imencode('.jpg', img)[1].tobytes()
        label = int(labels_dict[fields[2]])
        ex = make_example(img_raw, label, height, width)
        writer.write(ex.SerializeToString())
        #每写入1000条记录，向父进程发送消息，报告进度
        if file_num%10000==0:
            queue.put((pid, file_num))
        if file_num%max_num==0:
            writer.close()
            tfrecords_file_num += 1
            writer = tf.python_io.TFRecordWriter(targetfolder+"data_"+str(tfrecords_file_num)+".tfrecord")
    writer.close()        
    queue.put((pid, file_num))
 
#这个函数用来多进程生成TFRECORD文件，第一个参数是要处理的图片的文件名列表，第二个参数是需要用的CPU核心数
#第三个参数写入的文件目录名
def process_in_queues(fileslist, cores, targetfolder):
    total_files_num = len(fileslist)
    each_process_files_num = int(total_files_num/cores)
    print('############', 'actual number of images: ', each_process_files_num*cores, '############')
    #80个cpu, 差不多1281120
    files_for_process_list = []
    for i in range(cores):
        files_for_process_list.append(fileslist[i*each_process_files_num:(i+1)*each_process_files_num])
    files_number_list = [len(l) for l in files_for_process_list]
    
    each_process_tffiles_num = math.ceil(each_process_files_num/max_num)
    
    queues_list = []
    processes_list = []
    for i in range(cores):
        queues_list.append(Queue())
        #queue = Queue()
        processes_list.append(Process(target=gen_tfrecord, 
                                      args=(files_for_process_list[i],targetfolder,
                                      each_process_tffiles_num*i+1,queues_list[i],)))
 
    for p in processes_list:
        Process.start(p)
 
    #父进程循环查询队列的消息，并且每0.5秒更新一次
    while(True):
        try:
            total = 0
            progress_str=''
            for i in range(cores):
                msg=queues_list[i].get()
                total += msg[1]
                progress_str+='PID'+str(msg[0])+':'+str(msg[1])+'/'+ str(files_number_list[i])+'|'
            progress_str+='\r'
            #这个打印很厉害
            print(progress_str, end='')
            if total == each_process_files_num*cores:
            #if total == files_for_process_list*cores:
            #if total == total_files_num:    
                for p in processes_list:
                    p.terminate()
                    p.join()
                break
            time.sleep(0.5)
        except:
            break
    return total
 
if __name__ == '__main__':
    print('Start processing train data using %i CPU cores:'%cores)
    starttime=time.time()       	  
    total_processed = process_in_queues(train_images_name_list, cores, targetfolder="train_")
    endtime=time.time()
    print('\nProcess finish, total process %i images in %i seconds'%(total_processed, int(endtime-starttime)))
    
    print('************************************************************************')
    
    print('Start processing validation data using %i CPU cores:'%cores)
    starttime=time.time()  
    total_processed = process_in_queues(valid_images_name_list, cores, targetfolder="valid_")
    endtime=time.time()
    print('\nProcess finish, total process %i images, using %i seconds'%(total_processed, int(endtime-starttime)))
