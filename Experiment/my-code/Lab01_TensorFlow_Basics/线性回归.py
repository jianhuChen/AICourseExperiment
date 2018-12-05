#coding=utf-8
# File    : 1. 线性回归.py
# Desc    :
# Author  : jianhuChen
# license : Copyright(C), USTC
# Time    : 2018/10/11 19:37

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy随机生成100个随机点
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.1, 0.2], x_data) + 0.3

# print(x_data)

# 构造一个线性模型
b = tf.Variable(0.)
W = tf.Variable(np.float32(np.zeros([1, 2]))) #tf.random_uniform([1, 2], -1.0, 1.0)
y = tf.matmul(W, x_data) + b



# 二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))
# 定义一个梯度下降来进行训练的优化器 learning_rate = 0.2
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 最小化代价函数
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for step in range(201):
		sess.run(train)
		if step%20 == 0:
			print("step:" + str(step) + "\t"+ str(sess.run(W)) + "\t" + str(sess.run(b)))
