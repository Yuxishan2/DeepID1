#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 07:59:00 2018

@author: qingchuandong
"""

import tensorflow as tf
import numpy as np
import cv2

def read_image(image_path):
    image = cv2.imread(image_path)
    return np.asanyarray(image,dtype='uint8')

#def read_csv_file(csv_path):
#    images = []
#    labels = []
#    with open(csv_path,'r') as f:
#        for line in f.readlines():
#            image_path,label = line.split(',')
#            images.append(read_image(image_path))
#            labels.append(label)
#    
#    return np.asanyarray(images,dtype='float32'),np.asanyarray(labels,dtype='float32')
    
def write_tfrecord(csv_path,intend):
    writer = tf.python_io.TFRecordWriter(intend + '.tfrecord')

    with open(csv_path,'r') as f:
        for line in f.readlines():
            image_path,label = line.split()
            image_raw = read_image(image_path).tostring()
            example = tf.train.Example(
                    features = tf.train.Features(
                    feature={'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [int(label)])),
                             'image':tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_raw]))}))
            serialized = example.SerializeToString()
            writer.write(serialized)
    f.close()
    writer.close()
    return True

def write_tfrecord_pair(csv_path,intend):
    writer = tf.python_io.TFRecordWriter(intend + '.tfrecord')

    with open(csv_path,'r') as f:
        for line in f.readlines():
            image_path1,image_path2,label = line.split()
            image_raw1 = read_image(image_path1).tostring()
            image_raw2 = read_image(image_path1).tostring()
            example = tf.train.Example(
                    features = tf.train.Features(
                    feature={'label':tf.train.Feature(int64_list = tf.train.Int64List(value = [int(label)])),
                             'image1':tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_raw1])),
                             'image2':tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_raw2]))}))
            serialized = example.SerializeToString()
            writer.write(serialized)
    f.close()
    writer.close()
    return True

#def read_csv_pair_file(csv_path):
#    images1 = []
#    images2 = []
#    labels = []
#    with open(csv_path,'r') as f:
#        for line in f.readlines():
#            image_path1,image_path2,label = line.split(',')
#            images1.append(read_image(image_path1))
#            images2.append(read_image(image_path1))
#            labels.append(label)
#    
#    return np.asanyarray(images1,dtype='float32'),np.asanyarray(images1,dtype='float32'),np.asanyarray(labels,dtype='float32')

if __name__=="__main__":
    train_path = 'data/train_set.csv'
    valid_path = 'data/valid_set.csv'
    test_path = 'data/test_set.csv'
    
#    write_train_tfrecord = write_tfrecord(train_path,'train')
    
#    write_valid_tfrecord = write_tfrecord(valid_path,'valid')
    write_test_tfrecord = write_tfrecord_pair(test_path,'test')

