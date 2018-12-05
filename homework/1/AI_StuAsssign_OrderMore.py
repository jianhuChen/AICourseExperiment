import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split

def make_hparam_string(OrderNum, lr, BatchNum, CV, DataScaleFlag):
    return('Order_'+str(OrderNum)+'_lr_'+str(lr)+'_BatchNum_'+str(BatchNum)+'_CV_'+str(TestingDataRatio)+'_Flag_'+str(DataScaleFlag))

def GenInputMatrix(x,n):
    i = 2
    y = x
    while i<=n:
        y = np.hstack((np.power(x,i), y)) # 水平堆叠
        i = i+1
    return(y)

def DataScaling(X_train, X_test, flag):
    if flag == 1:
        E_X = np.sum(X_train,0) / X_train.shape[0] #求均值
        E_X.shape = 1,-1
        D_X = np.sqrt(np.sum(np.square(X_train - np.tile(E_X,[X_train.shape[0],1])),0)) / X_train.shape[0]
        X_train_Norm = (X_train - np.tile(E_X,[X_train.shape[0],1])) / np.tile(D_X,[X_train.shape[0],1])
        X_test_Norm = (X_test - np.tile(E_X, [X_test.shape[0], 1])) / np.tile(D_X,[X_test.shape[0],1])
        return (X_train_Norm,X_test_Norm,E_X,D_X)
    else:
        return (X_train, X_test, np.zeros([1, X_train.shape[1]]), np.ones([1, X_train.shape[1]]))

def Predict(X,W,b):
    # Hypothesis:Linear model
    b=b.reshape(1,1)
    b_expand = np.tile(b, [X.shape[0], 1])
#    print(W,b)
#    print(X.shape,b_expand.shape)
    return(np.matmul(X, W)+b_expand)

# Hyperparameters
training_epochs = int(2000)
#learning rate lr
lr = 1*1e-3
#total sample number N
N = 50
#TestingSamplesNumber/TotalSamplesNumber:TestingDataRatio
TestingDataRatio = 0.8
#the highest order of linear model:OrderNumcd
OrderNum = 3
#Use Student Number as seed
Seed = 16225151
#DataScalingFlag=1 indicates to use data scaling before fitting
DataScaleFlag = 1

BatchSize = N*(1-TestingDataRatio)
BatchNum = int(N * (1-TestingDataRatio)/BatchSize)
Train_LogDir = "C:/PyLogs/LinearRegress/Train_GD_" + make_hparam_string(OrderNum,lr,BatchNum,TestingDataRatio,DataScaleFlag)
Test_LogDir = "C:/PyLogs/LinearRegress/Test_GD_" + make_hparam_string(OrderNum,lr,BatchNum,TestingDataRatio,DataScaleFlag)
#Train_LogDir = "C:/PyLogs/LinearRegress/Train_Adam_" + make_hparam_string(OrderNum,lr,BatchNum,TestingDataRatio,DataScaleFlag)
#Test_LogDir = "C:/PyLogs/LinearRegress/Test_Adam_" + make_hparam_string(OrderNum,lr,BatchNum,TestingDataRatio,DataScaleFlag)

np.random.seed(Seed)
tf.set_random_seed(Seed)
#Construct N data samples(including training data set and testing data set)
x = np.linspace(0, 6, N) + np.random.randn(N)
x = np.sort(x)
y = x ** 2 - 4 * x - 3 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1

# print(x,y)
# print(x.shape, y.shape)

#training dataset and testing dataset: x -> [x**2 x]
x_expand = GenInputMatrix(x,OrderNum)
# print(x[0])
# print(x_expand[0])

# 划分训练集和测试集
x_train0, x_test0, y_train, y_test = train_test_split(x_expand, y, test_size = TestingDataRatio, random_state=Seed)
print(x_train0.shape, x_test0.shape, y_train.shape, y_test.shape)

#Input Data Scaling for training and testing procedure
[x_train, x_test, x_expand_E, x_expand_D] = DataScaling(x_train0, x_test0, DataScaleFlag)
# print(x_train0.shape)
# print(x_expand_E, x_expand_D) # 均值和标准差

# placeholders for a tensor of a Linear model with 2-order.
X = tf.placeholder(tf.float32, shape=[None, OrderNum])
Y = tf.placeholder(tf.float32, shape=[None, 1])

with tf.name_scope("Parameters") as scope:
    W = tf.Variable(tf.random_normal([OrderNum, 1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis:Linear model
with tf.name_scope("Hypothesis") as scope:
    hypothesis = tf.matmul(X, W) + b

# cost/loss function
with tf.name_scope("Cost") as scope:
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    Cost_summ = tf.summary.scalar("Batch_Cost", cost)
    Total_cost = tf.reduce_mean(tf.square(hypothesis - Y))
    Total_cost_summ = tf.summary.scalar("Total_cost", Total_cost)

# Minimize
with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum = 0.9)
    # optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum = 0.9)
    train = optimizer.minimize(cost)

# Launch the graph in a session.
with tf.Session() as sess:
    merged_summary = tf.summary.merge_all() # 将之前定义的所有summary op整合到一起
    Train_writer = tf.summary.FileWriter(Train_LogDir) #创建一个file writer用来向硬盘写summary数据,
    Test_writer = tf.summary.FileWriter(Test_LogDir)
    Train_writer.add_graph(sess.graph)  # Show the graph
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())
    print('Initial w,b:', sess.run([W, b]))
    print('TrainingDataNum|TestingDataNum=', N * (1-TestingDataRatio), '|', N * TestingDataRatio)
    print("fit with "+str(OrderNum)+"-order linear model...")

    # 训练之前的cost，可以与训练之后进行对比
    prev_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
    Initial_test_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})
    print("prev_cost = " + str(prev_cost) + " Initial_test_cost = " + str(Initial_test_cost))

	# 用一个列表存储Train_cost和Test_cost
    Train_cost_history = [prev_cost]
    Test_cost_history = [Initial_test_cost]
    OuterLoopFlag = 0
    Train_writer.add_summary(sess.run(Total_cost_summ, feed_dict={X: x_train, Y: y_train}), 0)
    Test_writer.add_summary(sess.run(Total_cost_summ, feed_dict={X: x_test, Y: y_test}), 0)
    time_start = time.time()
    for epoch in range(training_epochs):
       for j in range(BatchNum):
            StartIndex = int(j*BatchSize)
            EndIndex = int((j+1) * BatchSize)
            Batch_error, _ = sess.run([cost, train], feed_dict={X: x_train[StartIndex:EndIndex,:], Y: y_train[StartIndex:EndIndex,:]})
            Train_writer.add_summary(sess.run(Cost_summ, feed_dict={X: x_train, Y: y_train}), BatchNum*epoch+j)
            Test_writer.add_summary(sess.run(Cost_summ, feed_dict={X: x_test, Y: y_test}), BatchNum*epoch+j)
            curr_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
            # print(prev_cost,curr_cost)
            Testing_cost = sess.run(cost, feed_dict={X: x_test, Y: y_test})
            Train_cost_history.append(curr_cost)
            Test_cost_history.append(Testing_cost)
            prev_cost = curr_cost
       Train_writer.add_summary(sess.run(Total_cost_summ, feed_dict={X: x_train, Y: y_train}), epoch+1)
       Test_writer.add_summary(sess.run(Total_cost_summ, feed_dict={X: x_test, Y: y_test}), epoch+1)
    W_star, b_star = sess.run([W, b])
time_end = time.time()
W_star = W_star / x_expand_D.reshape([-1,1])
b_star = b_star - np.sum(W_star * x_expand_E.reshape([-1,1]))
print(u'最佳参数 W,b:', W_star, b_star)
print(time_end - time_start, u'秒')
# Error report on Training and Testing
print('Training...Cost=', curr_cost)
print('Testing...Cost=', Testing_cost)

plt.figure(1)
plt.subplot(1, 2, 1)
#print(x_train0.shape)
plt.plot(x_train0[:, -1], y_train, 'ro', ms=5, label='training data')
plt.plot(x_test0[:, -1], y_test, 'go', ms=5, label='testing data')
x1 = np.linspace(-2, 10, 1000)
x1.shape = -1, 1
x1_expand = GenInputMatrix(x1, OrderNum)
# print(x1_expand.shape, W_star.shape, b_star.shape)
Est_y1 = Predict(x1_expand, W_star, b_star)
plt.plot(x1, Est_y1, "b--", label='Fitting curve')
plt.legend(loc='upper left')
plt.grid(True)
plt.title("fit with "+str(OrderNum)+"-order linear model", fontsize=18)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)

plt.subplot(1, 2, 2)
NoParam = np.arange(0, len(Train_cost_history), 1)
#print(NoParam, cost_history)
plt.plot(NoParam, Train_cost_history, 'r-', ms=10,label='Training Learning Curve')
plt.plot(NoParam, Test_cost_history, 'b-', ms=10,label='Testing Learning Curve')
plt.xlabel('#PramUpdate', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curve', fontsize=18)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
