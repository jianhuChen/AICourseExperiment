#coding=utf-8
# File    : captcha_model.py
# Desc    :
# Author  : https://cloud.tencent.com/developer/labs/lab/10330
# Time    : 2018/11/1 18:37

# -*- coding: utf-8 -*
import tensorflow as tf
import math

class captchaModel():
	def __init__(self,
					 width=160,
					 height=60,
					 char_num=4,
					 classes=62):
		self.width = width
		self.height = height
		self.char_num = char_num
		self.classes = classes

    # 卷积层
	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	# 池化
	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	# 初始化权值
	def weight_variable(self, shape):
		# 生成一个截断的正态分布
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	# 初始化偏置
	def bias_variable(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	#网络模型
	def create_model(self,x_images,keep_prob):
		#first layer
		# 初始化第1个卷积层的权值和偏置
		w_conv1 = self.weight_variable([5, 5, 1, 32])
		b_conv1 = self.bias_variable([32])
		# 把x_image和权值进行卷积，再加上偏置值，再应用于relu激活函数
		h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(x_images, w_conv1), b_conv1))
		h_pool1 = self.max_pool_2x2(h_conv1)
		#dropout
		h_dropout1 = tf.nn.dropout(h_pool1, keep_prob)
		conv_width = math.ceil(self.width/2)
		conv_height = math.ceil(self.height/2)

		#second layer
		# 初始化第2个卷积层的权值和偏置
		w_conv2 = self.weight_variable([5, 5, 32, 64])
		b_conv2 = self.bias_variable([64])
		# 把h_dropout1和权值进行卷积，再加上偏置值，再应用于relu激活函数
		h_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_dropout1, w_conv2), b_conv2))
		h_pool2 = self.max_pool_2x2(h_conv2)
		h_dropout2 = tf.nn.dropout(h_pool2,keep_prob)
		conv_width = math.ceil(conv_width/2)
		conv_height = math.ceil(conv_height/2)

		#third layer
		# 初始化第3个卷积层的权值和偏置
		w_conv3 = self.weight_variable([5, 5, 64, 64])
		b_conv3 = self.bias_variable([64])
		# 把h_pool2和权值进行卷积，再加上偏置值，再应用于relu激活函数
		h_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_dropout2, w_conv3), b_conv3))
		h_pool3 = self.max_pool_2x2(h_conv3)
		h_dropout3 = tf.nn.dropout(h_pool3,keep_prob)
		conv_width = math.ceil(conv_width/2)
		conv_height = math.ceil(conv_height/2)

		# 160*60的图片第1次卷积后还是160*60（padding='SAME'）,第1次池化后变成了80*30
		# 第2次卷积后变成了80*30，第2次池化后变成了40*15
		# 第3次卷积后变成了40*15，第3次池化后变成了20*8
		# 经过上面的操作后得到64张20*8的平面

		#first fully layer
		conv_width = int(conv_width)
		conv_height = int(conv_height)
		# 初始化第1个全连接层的权值
		# 上一层有20*8*64个神经元，全连接层有1024个神经元
		w_fc1 = self.weight_variable([64*conv_width*conv_height, 1024])
		b_fc1 = self.bias_variable([1024])
		# 把池化层3的输出扁平化为1维
		h_dropout3_flat = tf.reshape(h_dropout3, [-1, 64*conv_width*conv_height])
		# 求第一个全连接层的输出
		h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_dropout3_flat, w_fc1), b_fc1))
		# Dropout: keep_prob用来表示神经元的输出概率
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		#second fully layer
		# 初始化第2个全连接层
		w_fc2 = self.weight_variable([1024, self.char_num*self.classes])
		b_fc2 = self.bias_variable([self.char_num*self.classes])
		y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)

		# [1, 40]
		return y_conv