#coding=utf-8
# File    : 3. 手写数字识别1.0.py
# Desc    : 使用逻辑回归完成多分类的手写数字识别
# Author  : jianhuChen
# license : Copyright(C), USTC
# Time    : 2018/10/11 20:09

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# train_images = mnist.train.images
# train_labels = mnist.train.labels
# test_images = mnist.train.images
# test_labels = mnist.train.labels
#
# print(train_images.shape)
# print(train_labels.shape)

# 每个批次的大小
batch_size = 100

# 定义两个placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# 权重和偏向都设为0
Weights = tf.Variable(tf.zeros(shape=[784, 10]))
biases = tf.Variable(tf.zeros(shape=[10]))
Wx_plus_b = tf.matmul(x, Weights) + biases

# 用softmax构建逻辑回归模型 非线性激活函数 多分类问题
prediction = tf.nn.softmax(Wx_plus_b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
# 交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 梯度下降算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 求正确率
# 结果存放在一个布尔列表中 argmax返回一维张量中最大值所在的位置索引
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(25):
		avg_cost = 0.
		for batch in range(n_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
			# 计算损失平均值
			avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}) / n_batch

		if (epoch+1)%5 == 0:
			acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
			print("epoch:" + str(epoch + 1) + " Testing accuracy = " + str(acc) + " Cost = " + str(avg_cost))
	print("="*25+"\n运行完成.")


# LossFunction: loss = tf.reduce_mean(tf.square(y-prediction)) learning_rate = 0.1
# epoch:5 Testing accuracy = 0.877 Cost = 0.025913709803399684
# epoch:10 Testing accuracy = 0.8946 Cost = 0.02028368369083514
# epoch:15 Testing accuracy = 0.9008 Cost = 0.018197474220598298
# epoch:20 Testing accuracy = 0.9057 Cost = 0.017018919846212326
# epoch:25 Testing accuracy = 0.9076 Cost = 0.016228531831028802

# LossFunction: loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# learning_rate = 0.1
# epoch:5 Testing accuracy = 0.8984 Cost = 1.6158899445967232
# epoch:10 Testing accuracy = 0.9087 Cost = 1.5867477403987547
# epoch:15 Testing accuracy = 0.9127 Cost = 1.5757358368960315
# epoch:20 Testing accuracy = 0.9165 Cost = 1.5692915992303322
# epoch:25 Testing accuracy = 0.9187 Cost = 1.5648277499459013

# LossFunction: loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# learning_rate = 0.01
# epoch:5 Testing accuracy = 0.7905 Cost = 1.8331100934202025
# epoch:10 Testing accuracy = 0.8121 Cost = 1.7407750084183433
# epoch:15 Testing accuracy = 0.82 Cost = 1.710426997704939
# epoch:20 Testing accuracy = 0.825 Cost = 1.6941950724341635
# epoch:25 Testing accuracy = 0.8284 Cost = 1.6828776806051082