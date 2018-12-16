# 引入Numpy
import numpy as np

# 构造一个线性回归函数
# y = W * x + b
W = 0.6
b = 0.4


#生成测试数据
def get_train_data(data_lenght):
    train_arr = []
    for i in range(data_lenght):
        tr_x = np.random.uniform(0.0, 1.0)
        tr_y = tr_x * W + b + np.random.uniform(-0.02, 0.02)
        train_arr.append([tr_x,tr_y])

    return train_arr


#生成校验数据
def get_validate_data(data_lenght):
    validate_arr = []
    for i in range(data_lenght):
        va_x = np.random.uniform(0.0, 1.0)
        va_y = va_x * W + b + np.random.uniform(-0.02, 0.02)
        validate_arr.append([va_x,va_y])

    return validate_arr

