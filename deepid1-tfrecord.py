#! /usr/bin/python
#import numpy as np
import tensorflow as tf

class_num = 1282

def weight_variable(shape):
    with tf.name_scope('weights'):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    with tf.name_scope('biases'):
        return tf.Variable(tf.zeros(shape))

def Wx_plus_b(weights, x, biases):
    with tf.name_scope('Wx_plus_b'):
        return tf.matmul(x, weights) + biases

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        weights = weight_variable([input_dim, output_dim])
        biases = bias_variable([output_dim])
        preactivate = Wx_plus_b(weights, input_tensor, biases)
        if act != None:
            activations = act(preactivate, name='activation')
            return activations
        else:
            return preactivate

def conv_pool_layer(x, w_shape, b_shape, layer_name, act=tf.nn.relu, only_conv=False):
    with tf.name_scope(layer_name):
        W = weight_variable(w_shape)
        b = bias_variable(b_shape)
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID', name='conv2d')
        h = conv + b
        relu = act(h, name='relu')
        if only_conv == True:
            return relu
        pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max-pooling')
        return pool

def accuracy(y_estimate, y_real):
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):  
            correct_prediction = tf.equal(tf.argmax(y_estimate,1), tf.argmax(y_real,1))
        with tf.name_scope('accuracy'):  
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)  
        return accuracy





if __name__ == '__main__':

    batch_size = 128
    
    tfrecords_train_filename = "/home/qingchuandong/project/face_deep/DeepID1-master/train.tfrecord"
    tfrecords_valid_filename = "/home/qingchuandong/project/face_deep/DeepID1-master/valid.tfrecord"
    train_filename_queue = tf.train.string_input_producer(['train.tfrecord'], num_epochs=200)
#    valid_filename_queue = tf.train.string_input_producer(['valid.tfrecord'], num_epochs=None)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(train_filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['image'],tf.uint8)
    image = tf.reshape(image, [55,47,3])
    label = tf.one_hot(features['label'],depth=class_num)
#    label = tf.cast(label, tf.int64)
#    image, label = tf.train.shuffle_batch([image, label],batch_size=batch_size, 
#                                                         num_threads=12, 
#                                                         capacity=896,
#                                                         min_after_dequeue=512)

    image, label = tf.train.shuffle_batch([image, label],batch_size=256, 
                                                         num_threads=8, 
                                                         capacity=1024,
                                                         min_after_dequeue=200)
    
#    with tf.name_scope('input'):
#        h0 = tf.placeholder(tf.float32, [None, 55, 47, 3], name='x')
#        y_ = tf.placeholder(tf.float32, [None, class_num], name='y')

    image = tf.cast(image, tf.float32)
    h1 = conv_pool_layer(image, [4, 4, 3, 20], [20], 'Conv_layer_1')
    h2 = conv_pool_layer(h1, [3, 3, 20, 40], [40], 'Conv_layer_2')
    h3 = conv_pool_layer(h2, [3, 3, 40, 60], [60], 'Conv_layer_3')
    h4 = conv_pool_layer(h3, [2, 2, 60, 80], [80], 'Conv_layer_4', only_conv=True)
    
    with tf.name_scope('DeepID1'):
        h3r = tf.reshape(h3, [-1, 5*4*60])
        h4r = tf.reshape(h4, [-1, 4*3*80])
        W1 = weight_variable([5*4*60, 160])
        W2 = weight_variable([4*3*80, 160])
        b = bias_variable([160])
        h = tf.matmul(h3r, W1) + tf.matmul(h4r, W2) + b
        h5 = tf.nn.relu(h)
    
    with tf.name_scope('loss'):
        y = nn_layer(h5, 160, class_num, 'nn_layer', act=None)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y,labels = label))
        tf.summary.scalar('loss', loss)
    
    accuracy = accuracy(y, label)
    global_step = tf.Variable(0,trainable = False)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss,global_step=global_step)
    
    merged = tf.summary.merge_all()  
    saver = tf.train.Saver()
#    data_x = trainX
#    data_y = (np.arange(class_num) == trainY[:,None]).astype(np.float32)
#    validY = (np.arange(class_num) == validY[:,None]).astype(np.float32)
    
    logdir = 'log'
    if tf.gfile.Exists(logdir):
        tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)
    
    sess = tf.Session()
    
    with sess.as_default():
#        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(logdir + '/train', sess.graph)
        sess.run(tf.initialize_all_variables())
        sess.run(tf.initialize_local_variables())
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        
        
        try:
            while not coord.should_stop():
                summary,_ = sess.run([merged,train_step])
#                print(sess.run(global_step))
                train_writer.add_summary(summary, global_step=sess.run(global_step))
                if sess.run(global_step) % 100 == 0:
                    print(sess.run(loss))
                if sess.run(loss) < 1.0 :
                    coord.request_stop()
                if sess.run(global_step) % 3000 == 0 and sess.run(global_step)!=0:
                    saver.save(sess,'checkpoint_2/mymodel',global_step=sess.run(global_step))
        except tf.errors.OutOfRangeError:
            print( 'Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
    
    

