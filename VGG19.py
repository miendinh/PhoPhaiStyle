import tensorflow as tf
import numpy as np
import sys

class VGG19:
    def __init__(self):
        None

    def conv2d(self):

        self.parameters = []

        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([127.5], dtype=tf.float32, shape=[1, 1, 1, 1], name='img_mean')
            images = self.X - mean

        #0
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                                 name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv1_1 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv1_1.kernel', kernel)

        #2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                                 name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv1_2 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv1_2.kernel', kernel)

        self.pool1 = tf.nn.max_pool(self.conv1_1,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')
        #5
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1),
                                 name='weights')
            conv = tf.nn.conv2d(conv1_2, kernel, [1, 1, 1, 1], padding='SAME')

            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv2_1.kernel', kernel)

        #7
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1),
                                 name='weights')
            conv = tf.nn.conv2d(conv1_2, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv2_2 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv2_2.kernel', kernel)

        #9
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')
        #10
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv3_1.kernel', kernel)

        #12
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv3_2 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv3_2.kernel', kernel)

        #14
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv3_3 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv3_3.kernel', kernel)

        #16
        with tf.name_scope('conv3_4') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv3_3, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv3_4 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv3_4.kernel', kernel)

        self.pool3 = tf.nn.max_pool(self.conv3_4, ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        #19
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv4_1 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv4_1.kernel', kernel)

        #21
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv4_2 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv4_2.kernel', kernel)

        #23
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv4_3 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv4_3.kernel', kernel)

        #25
        with tf.name_scope('conv4_4') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv4_3, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv4_4 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv4_4.kernel', kernel)

        #27
        self.pool4 = tf.nn.max_pool(self.conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        #28
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv5_1 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv5_1.kernel', kernel)

        #30
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv5_2 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv5_2.kernel', kernel)

        #32
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv5_3 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv5_3.kernel', kernel)

        #34
        with tf.name_scope('conv5_4') as scope:
            kernel = tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=1e-1), dtype=tf.float32, name='weights')
            conv = tf.nn.conv2d(self.conv5_3, kernel, [1, 1, 1, 1], padding='SAME')
            self.conv5_4 = tf.nn.relu(conv, name=scope)
            self.parameters += [kernel]
            if self.log:
                tf.summary.histogram('conv5_4.kernel', kernel)

        #36
        self.pool5 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')


    def fc_layers(self):
        #37
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.get_variable("fc1w", shape=[7*7*512, 4096], initializer=tf.contrib.layers.xavier_initializer())
            dropout3_flat = tf.reshape(self.pool5, [-1, shape])
            fc1 = tf.matmul(dropout3_flat, fc1w)
            self.fc1 = tf.nn.relu(fc1)
            self.parameters += [fc1w]
            if self.log:
                tf.summary.histogram('fc1.weights', fc1w)

        #38
        with tf.name_scope('fc2') as scope:
            fc2w = tf.get_variable("fc2w", shape=[1*1*4096, 4096)], initializer=tf.contrib.layers.xavier_initializer())
            self.fc2 = tf.matmul(self.dropout_fc1, fc2w, name="logits")
            self.parameters += [fc2w]
            if self.log:
                tf.summary.histogram('fc2.weights', fc2w)

        #41
        with tf.name_scope('fc3') as scope:
            fc3w = tf.get_variable("fc3w", shape=[1*1*4096, 1000], initializer=tf.contrib.layers.xavier_initializer())
            self.fc3 = tf.matmul(self.fc3, fc2w)
            self.parameters += [fc3w]
            if self.log:
                tf.summary.histogram('fc3.weights', fc2w)
