#coding=utf-8
# File    : generate_testing_set.py
# Desc    :
# Author  : jianhuChen
# license : Copyright(C), USTC
# Time    : 2018/10/31 10:44

from generate_captcha import generateCaptcha

captcha = generateCaptcha()

for i in range(100):
	captcha.gen_test_captcha('TestingSet')

print("Complete.........")