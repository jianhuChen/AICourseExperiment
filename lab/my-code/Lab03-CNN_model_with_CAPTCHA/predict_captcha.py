#coding=utf-8
# File    : predict_captcha.py
# Desc    :
# Author  : https://cloud.tencent.com/developer/labs/lab/10330
# Time    : 2018/11/1 18:38

# 首先导入需要的库
from PIL import Image, ImageFilter
import os
import tensorflow as tf
import numpy as np
import string
import sys
import generate_captcha
import captcha_model

if __name__ == '__main__':
	captcha = generate_captcha.generateCaptcha()
	width, height, char_num, characters, classes = captcha.get_parameter()

	# 在执行这个文件时，需要指定一个测试集的目录
	# 然后再获取这个目录里的所有图片名字列表
	file_list = os.listdir(sys.argv[1]) # 获取此路径下的所有文件的名字列表
	# print(file_list)
	# 将100个训练样本打包到test_x
	test_x = []
	for file in file_list:
		gray_image = Image.open(sys.argv[1] + '/' + file).convert('L')
		img = np.array(gray_image.getdata())
		test_x.append(np.reshape(img, [height, width, 1])/255.0)

	# 定义一些placeholder作为输入数据的入口
	x = tf.placeholder(tf.float32, [None, height, width, 1])
	# 使用dropout时的因子
	keep_prob = tf.placeholder(tf.float32)
	# 创建一个我们之前设定好的模型
	# 后面再导入训练好的模型的参数即可使用这个模型啦
	model = captcha_model.captchaModel(width, height, char_num, classes)
	y_conv = model.create_model(x, keep_prob)
	# 计算预测值
	predict = tf.argmax(tf.reshape(y_conv, [-1,char_num, classes]),2)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	saver = tf.train.Saver()

	with tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)) as sess:
		sess.run(tf.global_variables_initializer()	)
		# 如果之前已经训练好了模型或者是训练了一半的模型则可以直接加载进来继续训练
		ckpt = tf.train.get_checkpoint_state(r'./')	
		if ckpt and ckpt.model_checkpoint_path:
			print(ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables
		else:
			# 如果没有找到保存的模型参数，打印出错信息
			print("Error：Model not found")
		# 开始喂入测试数据进行预测，结果保存到pre_list列表中
		pre_list =  sess.run(predict, feed_dict={x: test_x, keep_prob: 1})
		# 定义一个计数器，记录一共预测对了多少数字
		count = 0 
		for i, pre in enumerate(pre_list):
			s = ''
			for j, ch in enumerate(pre):
				s += characters[ch]
				# 如果预测的数字与真实的label相同则计数器+1
				if characters[ch] == file_list[i][j]:
					count += 1
			# 打印预测值与真实值
			print("Pre:", s, "\tLabel:", file_list[i][:-4])
		# 最后输出本次预测的正确率
		print('==================>accuracy:',count/(char_num*len(test_x)), '\t', count, '/', char_num*len(test_x))
