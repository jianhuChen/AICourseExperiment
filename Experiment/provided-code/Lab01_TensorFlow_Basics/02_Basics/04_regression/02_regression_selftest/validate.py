import matplotlib.pyplot as plt
#引入测试数据
from linearRegression import testData as pt

validateData = pt.get_validate_data(20)
va_x = [v[0] for v in validateData]
va_y = [v[1] for v in validateData]

#训练结果
y = []
for x in va_x :
    y.append(x * 0.3 + 0.8)


# 构造图形结构
plt.plot(va_x, va_y, 'ro', label='validate Data')
plt.plot(va_x,  y, label='train result')
plt.legend()
plt.show()