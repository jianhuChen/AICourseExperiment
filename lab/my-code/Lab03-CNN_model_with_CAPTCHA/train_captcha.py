#coding=utf-8
# File    : train_captcha.py
# Desc    :
# Author  : https://cloud.tencent.com/developer/labs/lab/10330
# Time    : 2018/11/1 18:30

#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import generate_captcha
import captcha_model

if __name__ == '__main__':
	# 调用之前定义的模块，产生一个能批量生成验证码的实例对象captcha
	captcha = generate_captcha.generateCaptcha()
	# 获取产生验证码时的一系列参数，后面会用得到
	width, height, char_num, characters, classes = captcha.get_parameter()

	# 定义3个placeholder，分别用来做特征输入，真实标签输入，dropout的程度
	x = tf.placeholder(tf.float32, [None, height, width, 1])
	y_ = tf.placeholder(tf.float32, [None, char_num*classes])
	keep_prob = tf.placeholder(tf.float32)

	# 传入验证码的相关参数来实例化一个模型，这里用到了之前定义的模块captcha_model
	model = captcha_model.captchaModel(width, height, char_num, classes)
	# y_conv是特征输入通过模型后的预测值
	y_conv = model.create_model(x, keep_prob)

	# 交叉熵代价函数
	cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv))
	# 使用AdamOptimizer进行优化
	train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

	# 之前定义的模型中，最后的输出值应该是shape=[1, 40]的一个向量
	# 为了方便预测的label与真实的label之间比较，我们把它reshap成
	# [验证码中的字符个数, 类别个数]这样的形状
	predict = tf.reshape(y_conv, [-1, char_num, classes]) #[64, 4, 10]
	# 类似的也把正确的label也reshap成[验证码中的字符个数, 类别个数]这样的形状
	real = tf.reshape(y_, [-1, char_num, classes]) #[64, 4, 10]
	# 计算一共预测对了多少个字符
	correct_prediction = tf.equal(tf.argmax(predict, 2), tf.argmax(real, 2))
	# 计算预测正确率
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 实例化一个saver方便后面保存模型的参数
	saver = tf.train.Saver()
	# 使用GPU预算更快, 我的GPU是GTX850M
	# 最后实测训练到acc>99%耗时30分钟左右
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	with tf.Session() as sess:
		# 初始化全局变量
		sess.run(tf.global_variables_initializer())

		# 如果之前已经训练好了模型或者是训练了一半的模型
		# 则可以直接加载进来继续训练
		ckpt = tf.train.get_checkpoint_state(r'./')
		if ckpt and ckpt.model_checkpoint_path:
			print(ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

		step = 1
		time_start = time.time()
		while True:
			batch_size = 64
			# 产生mini-batch=64的训练集数据
			batch_x, batch_y = next(captcha.gen_captcha(64))
			_, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
			# 记录一下每一轮的训练时间
			time_end = time.time()
			print('step:%d,loss:%f' % (step, loss), '......time:', time_end - time_start, u'秒')
			# 每满100轮就调用100个验证码来测试一下准确率
			if step % 100 == 0:
				batch_x_test, batch_y_test = next(captcha.gen_captcha(100))
				acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
				print('==========>step:%ds,accuracy:%f' % (step, acc))
				# 为了训练方便（可以分多次训练），我这里设置了每100轮保存一下模型
				saver.save(sess, r"./capcha_model.ckpt")
				# 当在验证机集上的正确率大于99%时，保存模型并退出训练
				if acc > 0.995:
					saver.save(sess, r"./capcha_model.ckpt")
					break
			step += 1