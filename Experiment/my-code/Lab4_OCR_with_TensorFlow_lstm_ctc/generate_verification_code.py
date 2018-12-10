# -*- coding: utf-8 -*-
# @File 	: generate_verification_code.py
# @Author 	: jianhuChen
# @Date 	: 2018-11-18 23:16:09
# @License 	: Copyright(C), USTC
# @Last Modified by  : jianhuChen
# @Last Modified time: 2018-12-09 17:11:27

import string
import numpy as np
import freetype
import copy
import random
import cv2
import os

class put_chinese_text(object):
	def __init__(self, ttf):
		self._face = freetype.Face(ttf)

	def draw_text(self, image, pos, text, text_size, text_color):
		self._face.set_char_size(text_size * 64)
		metrics = self._face.size
		ascender = metrics.ascender/64.0

		ypos = int(ascender)

		if not isinstance(text, str):
			text = text.decode('utf-8')
		img = self.draw_string(image, pos[0], pos[1]+ypos, text, text_color)
		return img

	def draw_string(self, img, x_pos, y_pos, text, color):
		prev_char = 0
		pen = freetype.Vector()
		pen.x = x_pos << 6   # div 64
		pen.y = y_pos << 6

		hscale = 1.0
		matrix = freetype.Matrix(int(hscale)*0x10000, int(0.2*0x10000),\
								 int(0.0*0x10000), int(1.1*0x10000))
		cur_pen = freetype.Vector()
		pen_translate = freetype.Vector()

		image = copy.deepcopy(img)
		for cur_char in text:
			self._face.set_transform(matrix, pen_translate)

			self._face.load_char(cur_char)
			kerning = self._face.get_kerning(prev_char, cur_char)
			pen.x += kerning.x
			slot = self._face.glyph
			bitmap = slot.bitmap

			cur_pen.x = pen.x
			cur_pen.y = pen.y - slot.bitmap_top * 64
			self.draw_ft_bitmap(image, bitmap, cur_pen, color)

			pen.x += slot.advance.x
			prev_char = cur_char

		return image

	def draw_ft_bitmap(self, img, bitmap, pen, color):
		x_pos = pen.x >> 6
		y_pos = pen.y >> 6
		cols = bitmap.width
		rows = bitmap.rows

		glyph_pixels = bitmap.buffer

		for row in range(rows):
			for col in range(cols):
				if glyph_pixels[row*cols + col] != 0:
					img[y_pos + row][x_pos + col][0] = color[0]
					img[y_pos + row][x_pos + col][1] = color[1]
					img[y_pos + row][x_pos + col][2] = color[2]


class generateVerificationCode(object):
	def __init__(self,
					width=256, # 验证码图片的宽
					height=32, # 验证码图片的高
					char_max_size=5, # 验证码最多的字符个数
					characters=string.digits):# + string.ascii_uppercase + string.ascii_lowercase):#验证码组成，数字+大写字母+小写字母
		self.width = width
		self.height = height
		self.char_max_size = char_max_size
		self.characters = characters
		self.classes = len(characters) # 一共有多少类
		self.char_set = list(self.characters)
		self.ft = put_chinese_text('fonts/OCR-B.ttf')

	def gen_verification_code(self, is_random=False, batch_size=50):
		X = np.zeros([batch_size, self.height, self.width, 3]) # 3个通道
		Y = np.zeros([batch_size, self.char_max_size, self.classes]) # one_hot编码
		text_list=[]
		while True:
			for i in range(batch_size):
				# 随机设定长度
				if is_random == True:
					size = random.randint(1, self.char_max_size)
				else:
					size = self.char_max_size
				# 产生size长度的随机串
				text = ''.join(random.sample(self.characters, size))
				# 保存所有str类型的label
				text_list.append(list(text))
				# 产生一张图片
				img = np.zeros([self.height, self.width, 3]) # 三个通道
				color_ = (255,255,255) # Write
				pos = (0, 3)
				text_size = 25
				image = self.ft.draw_text(img, pos, text, text_size, color_)
				X[i] = np.reshape(image, [self.height, self.width, 3]) # /255.0  # 将数据全部变换到 [0,1] 范围内
				for j, ch in enumerate(text):
					Y[i, j, self.characters.find(ch)] = 1	
				# print("X.shape = ", X.shape)
				# print(X)
				# print("text_list = ", text_list)
				# print("Y.shape = ", Y.shape)
				# print(Y)
			yield X, text_list, Y

	def gen_test_verification_code(self, dir, is_random=False, num=100):
		if not os.path.exists(dir):
			os.makedirs(dir)
		X = np.zeros([self.height, self.width, 3]) # 3个通道
		for i in range(num):
			# 随机设定长度
			if is_random == True:
				size = random.randint(1, self.char_max_size)
			else:
				size = self.char_max_size
			# 产生size长度的随机串
			text = ''.join(random.sample(self.characters, size))
			# 产生一张图片
			img = np.zeros([self.height, self.width, 3]) # 三个通道
			color_ = (255,255,255) # Write
			pos = (0, 3)
			text_size = 25
			image = self.ft.draw_text(img, pos, text, text_size, color_)
			X = np.reshape(image, [self.height, self.width, 3]) # /255.0  # 将数据全部变换到 [0,1] 范围内
			cv2.imwrite(dir+'/'+text+'.png', X[:, :, 2])
			print("\rGenerating.........(%d/%d)"%(i+1, num), end='')	
		print("\nTest verification code generated")


if __name__ == '__main__':
	genObj = generateVerificationCode()
	X, codes, Y = next(genObj.gen_verification_code(is_random=True, batch_size=3))
	for i in range(3):
		cv2.imshow(str(codes[i]), X[i])
	cv2.waitKey(0)

