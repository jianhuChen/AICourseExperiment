#coding=utf-8
# File    : 1. 线性回归v2.py
# Desc    :
# Author  : jianhuChen
# license : Copyright(C), USTC
# Time    : 2018/10/11 19:37
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy随机生成100个随机点
x_data = np.float32(np.random.rand(1, 100))
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = x_data*0.2 + 0.3 + noise

# print(x_data)

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 构造一个线性模型
b = tf.Variable(0.)
w = tf.Variable(0.)
y = w*x_data + b

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
			print("step:" + str(step) + "\t"+ str(sess.run([w, b])))
	# 获得预测值
	prediction_value = sess.run(y)
	# 画图
	plt.figure()
	# 画出训练集
	plt.scatter4(x_data, y_data)
	plt.plot(x_data, prediction_value, 'r-', lw=5)
	plt.show()