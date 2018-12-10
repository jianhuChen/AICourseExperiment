# -*- coding: utf-8 -*-
# @File 	: generate_verification_code.py
# @Author 	: jianhuChen
# @Date 	: 2018-11-18 23:16:09
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2018-12-09 17:11:27

# 首先导入需要的库
import tensorflow as tf
import cv2
import sys
import os
from generate_verification_code import *
from train_model import *


if __name__ == '__main__':
	# 设置产生测试集文件夹的名字
	Testing_set_dir = 'TestingSet'
	obj = generateVerificationCode()
	# 生成验证码
	obj.gen_test_verification_code(Testing_set_dir, is_random=True, num=100)
	# 获取这个目录里的所有图片名字列表
	file_list = os.listdir(Testing_set_dir) # 获取此路径下的所有文件的名字列表
	# print(file_list)
	# 将100个训练样本打包到test_x
	test_x = np.zeros([len(file_list), obj.height, obj.width]) # 1个通道
	text_list = []
	for i, file in enumerate(file_list):
		image = cv2.imread(Testing_set_dir + '/' + file)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		test_x[i] = gray_image
		text = list(file)[:-4] # remove '.png'
		text_list.append(text)
	# [num, height, width]=>[num, width, height]
	test_x = np.transpose(test_x, (0, 2, 1)) 
	test_targets = sparse_tuple_from(text_list)
	test_seq_len = np.ones(test_x.shape[0]) * test_x.shape[1]

	# 定义一些placeholder作为输入数据的入口
	inputs = tf.placeholder(tf.float32, [None, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
	#targets是标签(稀疏矩阵的形式)，因为定义ctc_loss需要的稀疏矩阵
	targets = tf.sparse_placeholder(tf.int32)
	#1维向量 序列长度 [batch_size,]
	#是一个长度=BATCH_SIZE=64的一维向量,每个里面都是256，表示一个batch中每一张有256条输入序列  
	seq_len = tf.placeholder(tf.int32, [None])
	# 调用前面定义好的模型，并传入值
	logits = RNN(inputs, seq_len)
	# 解码，获得结果
	decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)


	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		# 载入模型
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
		ckpt_dir = "./ckpt_dir"
		if not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)
		ckpt = tf.train.get_checkpoint_state(ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(session, ckpt.model_checkpoint_path)
			print("Restore succcess! path:", ckpt.model_checkpoint_path)
		else:
			# 如果没有找到保存的模型参数，打印出错信息
			print("Error：Model not found")

		# 开始喂入测试数据进行预测，结果保存到pre_list列表中
		test_feed = {inputs: test_x, targets: test_targets, seq_len: test_seq_len}
		decoded_list = session.run(decoded[0], test_feed)

		original_list = decode_sparse_tensor(test_targets)
		detected_list = decode_sparse_tensor(decoded_list)

		true_numer = 0
		if len(original_list) != len(detected_list):
			print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
				  " test and detect length desn't match")
		print("T/F: original(length) <-------> detectcted(length)")
		for idx, number in enumerate(original_list):
			detect_number = detected_list[idx]
			hit = (number == detect_number)
			print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
			if hit:
				true_numer = true_numer + 1
		Accuracy = true_numer * 1.0 / len(original_list)
		print("Test Accuracy:", Accuracy)

	