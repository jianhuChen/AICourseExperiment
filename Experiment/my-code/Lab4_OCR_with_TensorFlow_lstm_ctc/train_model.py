# -*- coding: utf-8 -*-
# @File 	: generate_verification_code.py
# @Author 	: jianhuChen
# @Date 	: 2018-11-18 23:16:09
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2018-12-09 17:11:27

from generate_verification_code import *
import numpy as np
import time 
import os
import tensorflow as tf

class FError(Exception):
	pass

#定义一些常量
#图片大小，32 x 256
OUTPUT_SHAPE = (32,256)

# LSTM 循环体个数=64  层数=1
num_hidden = 64
num_layers = 1

obj = generateVerificationCode()
num_classes = obj.classes + 1 + 1  # 10位数字 + blank + ctc blank

#训练最大轮次
num_epochs = 10000

#初始化学习速率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9  # The learning rate decay factor

DIGITS='0123456789'
BATCHES = 10
BATCH_SIZE = 64 
TRAIN_SIZE = BATCHES * BATCH_SIZE


#转化一个序列列表为稀疏矩阵    
def sparse_tuple_from(sequences, dtype=np.int32):
	indices = []
	values = []
	for n, seq in enumerate(sequences):
		indices.extend(zip([n] * len(seq), range(len(seq))))
		values.extend(seq)
	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
	return indices, values, shape
	
def decode_sparse_tensor(sparse_tensor):
	decoded_indexes = list()
	current_i = 0
	current_seq = []
	for offset, i_and_index in enumerate(sparse_tensor[0]):
		i = i_and_index[0]
		if i != current_i:
			decoded_indexes.append(current_seq)
			current_i = i
			current_seq = list()
		current_seq.append(offset)
	decoded_indexes.append(current_seq)
	result = []
	for index in decoded_indexes:
		result.append(decode_a_seq(index, sparse_tensor))
	return result
	
def decode_a_seq(indexes, spars_tensor):
	decoded = []
	for m in indexes:
		str = DIGITS[spars_tensor[1][m]]
		decoded.append(str)
	return decoded

# 生成一个训练batch
def get_next_batch(batch_size=128, is_random=True):
	obj = generateVerificationCode()
	# X.shape=[batch_size, height, width, 3]
	# 获取一个batch的数据集
	# codes为一个列表，所有str类型的label
	X, text_list, Y = next(obj.gen_verification_code(is_random=is_random, batch_size=batch_size))
	# 这一步的作用：改变形状[batch_size, height, width, 1]=>[batch_size, height, width]=>[batch_size, width, height]
	X = np.transpose(np.reshape(X[:,:,:,2], [batch_size,OUTPUT_SHAPE[0],OUTPUT_SHAPE[1]]), (0, 2, 1))
	#targets转成稀疏矩阵 
	# 返回三个值，非零元素的位置信息，非零元素对应的值，稀疏矩阵的形状
	sparse_targets = sparse_tuple_from(text_list)
	#(batch_size,) sequence_length值都是256，最大划分列数
	seq_len = np.ones(X.shape[0]) * OUTPUT_SHAPE[1]
	return X, sparse_targets, seq_len 

def RNN(inputs, seq_len):
	#定义LSTM网络
	cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
	stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
	#output的shape=(64,256,64)(batch_size*256条*64个output)[batch_size,seq_len,cell.output_size]
	outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)
	shape = tf.shape(inputs)
	batch_s, max_timesteps = shape[0], shape[1]
	# 为方便矩阵运算，reshape一下outputs，outputs的size是(64*256,64)
	outputs = tf.reshape(outputs, [-1, num_hidden])
	# 初始化权值  生成一个截断的正态分布
	weights = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name="W")
	bias = tf.Variable(tf.constant(0., shape=[num_classes]), name="b")
	# 计算结果
	# result形状：[64*256, 12]
	logits = tf.matmul(outputs, weights) + bias
	# reshape之后logits的shape是(64,256,12) [batch_size,seq_len,num_classes]
	logits = tf.reshape(logits, [tf.shape(inputs)[0], -1, num_classes])
	#交换坐标轴，axis0和axis1互换，logits的shape是(256,64,12) [seq_len,batch_size,num_classes]
	logits = tf.transpose(logits, (1, 0, 2))
	return logits

def do_report():
	test_inputs,test_targets,test_seq_len = get_next_batch(BATCH_SIZE)
	test_feed = {inputs: test_inputs, targets: test_targets, seq_len: test_seq_len}
	decoded_list, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
	
	original_list = decode_sparse_tensor(test_targets)
	detected_list = decode_sparse_tensor(decoded_list)

	true_numer = 0
	if len(original_list) != len(detected_list):
		print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
			  " test and detect length desn't match")
		return -1
	print("T/F: original(length) <-------> detectcted(length)")
	for idx, number in enumerate(original_list):
		detect_number = detected_list[idx]
		hit = (number == detect_number)
		print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
		if hit:
			true_numer = true_numer + 1
	Accuracy = true_numer * 1.0 / len(original_list)
	print("Test Accuracy:", Accuracy)
	return Accuracy

if __name__ == '__main__':
	# 用来记录全局step，每更新一次参数就会+1
	global_step = tf.Variable(0, trainable=False)
	inputs = tf.placeholder(tf.float32, [None, OUTPUT_SHAPE[1], OUTPUT_SHAPE[0]])
	#targets是标签(稀疏矩阵的形式)，因为定义ctc_loss需要的稀疏矩阵
	targets = tf.sparse_placeholder(tf.int32)
	#1维向量 序列长度 [batch_size,]
	#是一个长度=BATCH_SIZE=64的一维向量,每个里面都是256，表示一个batch中每一张有256条输入序列  
	seq_len = tf.placeholder(tf.int32, [None])
	# 定义指数下降的学习率
	learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS, LEARNING_RATE_DECAY_FACTOR, staircase=True)
	# 调用前面定义好的模型，并传入值
	logits = RNN(inputs, seq_len)
	# 定义CTC损失函数 需要喂入数据：targets，logits，seq_len
	cost = tf.reduce_mean(tf.nn.ctc_loss(labels=targets, inputs=logits, sequence_length=seq_len))
	# 使用AdamOptimizer进行优化
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)
	# 解码，获得结果
	decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
	# 求准确率
	acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

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

		# 开始训练
		# 定义训练结束的标志，当正确率达到某一值时，可以通过设置它来提前结束训练
		train_end_flag = False
		start_time = time.time() # 记录时间
		for curr_epoch in range(num_epochs):
			print("Epoch.......", curr_epoch)
			train_cost =  0
			for batch in range(BATCHES):
				b_start_time = time.time()
				# 产生mini-batch=64的训练集数据
				train_inputs, train_targets, train_seq_len = get_next_batch(BATCH_SIZE)
				feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}
				train_acc, _, b_cost, steps,  = session.run([acc, optimizer, cost, global_step], feed)
				train_cost += b_cost
				print("Step:", steps, ", batch_time:", time.time()-b_start_time, 's')
				# 每REPORT_STEPS个step测试一下正确率，并打印预测结果和真实label
				if steps > 0 and steps % REPORT_STEPS == 0:
					if(do_report()>0.9):
						save_path = saver.save(session, ckpt_dir+"/model.ckpt", global_step=steps)
						print("save succcess! path:", save_path)
						print("=============================>Train succcess, Time:", time.time()-start_time, 's!')
						# 如果正确率超过了0.9就可以提起停止训练啦
						train_end_flag = True
						break
			# 提前停止训练            
			if train_end_flag:
				break
			# 每过一个epoch，计算一个训练误差
			train_cost /= BATCHES
			# 每过一个epoch，调用一下验证集测试
			val_inputs, val_targets, val_seq_len = get_next_batch(BATCH_SIZE)
			val_feed = {inputs: val_inputs,	targets: val_targets, seq_len: val_seq_len}
			val_cost, vald_acc, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
			log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, val_cost = {:.3f}, time = {:.3f}s, learning_rate = {}"
			print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, val_cost, time.time()-b_start_time, lr))