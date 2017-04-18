# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from read import ReadDataset
import time
import pickle


#参数设置
class Flags:
    def __init__(self):
        self.num_neurons = 1024  # 全连接层神经元数量
        self.label_num = 6 #分类问题，一共六个类
        self.train_dir = '../SICK/SICK.txt'#训练集所在位置
        self.input_width = 32#CNN输入大小
        self.input_height = 32
        self.input_color_channels = 4#CNN输入层数
        self.embeddings_dir = '../W2V/embeddings.pkl'
        self.batch_size = 50
        self.hidden_size = 128#仅仅是初始值，hidden_size由read_embeddings函数确定
        self.zero_state_fw = None
        self.zero_state_bw = None
        self.num_steps = 32

FLAGS = Flags()
#构建BiLSTM

lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.hidden_size)
state_bw = FLAGS.zero_state_bw = lstm_bw_cell.zero_state(FLAGS.batch_size, tf.float32)
state_fw = FLAGS.zero_state_fw = lstm_fw_cell.zero_state(FLAGS.batch_size, tf.float32)


#函数定义
def conv2d(x, shape): #x为输入
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]))
    conv = tf.nn.relu(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') + b)
    pool = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool

#构建CNN models
#Convolutional Layer
def cnn(x):
    pool1 = conv2d(x, [3, 3, FLAGS.input_color_channels, 128])
    pool2 = conv2d(pool1, [3, 3, 128, 164])
    pool3 = conv2d(pool2, [3, 3, 164, 192])
    pool4 = conv2d(pool3, [3, 3, 192, 192])
    pool5 = conv2d(pool4, [3, 3, 192, 128])
    pool5_flat = tf.reshape(pool5, [-1, 128])

    # Densely Connected Layer
    W_fc1 = tf.Variable(tf.truncated_normal([128, FLAGS.num_neurons], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[FLAGS.num_neurons]))
    h_fc1 = tf.nn.relu(tf.matmul(pool5_flat, W_fc1) + b_fc1)

    W_fc2 = tf.Variable(tf.truncated_normal([FLAGS.num_neurons,1], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[1]))
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv

def read_embeddings(embeddings_dir):
    read_data = open(embeddings_dir, 'rb')
    embeddings = pickle.load(read_data)
    read_data.close()
    lenght = len(embeddings[0])
    FLAGS.hidden_size = lenght
    embeddings[0] = [0. for i in range(lenght)]
    return embeddings

embeddings = read_embeddings(FLAGS.embeddings_dir)

#输入输出设置
s1 = tf.placeholder(tf.int32, [None, FLAGS.input_width])
s2 = tf.placeholder(tf.int32, [None, FLAGS.input_width])
similarity = tf.placeholder(tf.float32, [None])

#sim_label = tf.nn.embedding_lookup(one_hot, similarity) #need array or list return array

#训练 
with tf.Session() as sess:
    stc1 = tf.nn.embedding_lookup(embeddings, s1) #stc [batch_size, word_num, word2vec]
    stc2 = tf.nn.embedding_lookup(embeddings, s2)
    stc1 = tf.unstack(stc1, FLAGS.num_steps, 1) #word_num is num_steps
    stc2 = tf.unstack(stc2, FLAGS.num_steps, 1)

    with tf.variable_scope("RNN"):
        word_contexts1, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, stc1, state_fw, state_bw)
        tf.get_variable_scope().reuse_variables()
        word_contexts2, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, stc2, state_fw, state_bw)

    word_contexts1 = tf.transpose(word_contexts1, [1, 0, 2])
    word_contexts2 = tf.transpose(word_contexts2, [1, 0, 2])
    h1_for, h1_back = tf.split(word_contexts1,[FLAGS.hidden_size, FLAGS.hidden_size], 2)
    h2_for, h2_back = tf.split(word_contexts2, [FLAGS.hidden_size, FLAGS.hidden_size], 2)

    cos_dis0 = tf.matmul((h1_for + h1_back), (h2_for + h2_back), transpose_b=True)
    cos_dis1 = tf.matmul(h1_for, h2_for, transpose_b=True)
    cos_dis2 = tf.matmul(h1_back, h2_back, transpose_b=True)
    cos_dis3 = cos_dis1 + cos_dis2
    cos_dis = [cos_dis0, cos_dis1, cos_dis2, cos_dis3]
    cos_dis = tf.transpose(cos_dis,[1,2,3,0])
    input_layer = tf.reshape(cos_dis, [-1, 32, 32, 4])

    output = cnn(input_layer)

    loss = tf.reduce_mean(tf.square(similarity - output))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    init = tf.global_variables_initializer()
    init.run()

    train_dataset = ReadDataset(FLAGS.train_dir)
    start_time = time.time()
    sp = start_time

    for i in range(2000):
        x, y, z= train_dataset.next_batch(FLAGS.batch_size)
        if (i%100 == 0):
            l = sess.run(loss,feed_dict={s1: x, s2: y, similarity: z})
            point = time.time() - sp
            print("step %d, loss: %f, used time: %f(sec)"%(i, l, point))
            sp = time.time()
        train_step.run({s1: x, s2: y, similarity: z})
    # saver = tf.train.Saver()
    # saver.save(sess,'./logs/cnn.ckpt')
    # summarywriter = tf.summary.FileWriter('./logs', sess.graph)
    duration = time.time() - start_time
    print("total cost time:%f(sec)"%duration)
