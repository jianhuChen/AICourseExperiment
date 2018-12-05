#coding=utf-8
# File    : 4. 手写数字识别2.0.py
# Desc    : 使用人工神经网络完成手写数字识别 网络结构[784, 625, 10]
# Author  : jianhuChen
# license : Copyright(C), USTC
# Time    : 2018/10/11 20:49

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# 创建一个简单的神经网络
Weights_L1 = tf.Variable(tf.random_normal(shape=[784, 625], mean=0, stddev=1/tf.sqrt(784.)))
biases_L1 = tf.Variable(tf.zeros(shape=[625])+0.1)
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.sigmoid(Wx_plus_b_L1)

Weights_L2 = tf.Variable(tf.random_normal(shape=[625, 10], mean=0, stddev=1/tf.sqrt(625.)))
biases_L2 = tf.Variable(tf.zeros(shape=[10])+0.1)
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.softmax(Wx_plus_b_L2)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降算法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔列表中 argmax返回一维张量中最大值所在的位置索引
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(24):
		avg_cost = 0.
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
			# 计算损失平均值
			avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}) / n_batch

		if (epoch + 1) % 5 == 0:
			acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
			print("epoch:" + str(epoch + 1) + " Testing accuracy = " + str(acc) + " Cost = " + str(avg_cost))
	print("=" * 25 + "\n运行完成.")

# weight initial：tf.random_normal(shape=[784, 625], mean=0, stddev=1)
# epoch:5 Testing accuracy = 0.8514 Cost = 1.6187160979617716
# epoch:10 Testing accuracy = 0.9207 Cost = 1.5406855411963032
# epoch:15 Testing accuracy = 0.9336 Cost = 1.5271125626564013
# epoch:20 Testing accuracy = 0.9403 Cost = 1.5189318659088829
# epoch:25 Testing accuracy = 0.9447 Cost = 1.5125654257427577

# weight initial：tf.random_normal(shape=[784, 625], mean=0, stddev=1/tf.sqrt(784.))
# epoch:5 Testing accuracy = 0.8271 Cost = 1.6473344267498358
# epoch:10 Testing accuracy = 0.836 Cost = 1.6243998995694238
# epoch:15 Testing accuracy = 0.9252 Cost = 1.5357917167923671
# epoch:20 Testing accuracy = 0.933 Cost = 1.5267354731126248
