
# coding: utf-8

# # 全连接神经网络

# In[27]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


# 导入数据集函数
def loadData(path='mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'],f['y_train']
    x_test, y_test = f['x_test'],f['y_test']
    f.close()
    return (x_train,y_train),(x_test,y_test)


# In[28]:


# 从Keras导入Mnist数据集
(x_train, y_train), (x_validation, y_validation) = loadData()

# 显示4张手写数字图片
plt.subplot(221)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))

plt.subplot(222)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))

plt.subplot(223)
plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))

plt.subplot(224)
plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))

plt.show()


# In[29]:


# 设置图像输入的像素大小

num_pixels = x_train.shape[1] * x_train.shape[2]  # 28* 28 


# 设定随机种子
seed = 7
np.random.seed(seed)

# 原始数据中，每张图片大小为28*28的矩阵，因此将每张图片的像素值展开为一个数组，每个数组大小为28*28 = 784
# 训练集和验证集做同样的处理
x_train = x_train.reshape(x_train.shape[0],num_pixels).astype('float') 
x_validation = x_validation.reshape(x_validation.shape[0],num_pixels).astype('float')

# 格式化数据
x_train = x_train /255
x_validation = x_validation/255

#由于最终的输出结果是0-10的数字，属于多目标预测，因此进行one_hot编码，提高效率
y_train = np_utils.to_categorical(y_train)
y_validation = np_utils.to_categorical(y_validation) # y_validation 由一维矩阵转换为二维矩阵

# 设置输出类别：10，共有10种结果
num_classes = y_validation.shape[1]



# 定义多层感知机模型：MLP模型
def create_model():
    model = Sequential()
    model.add(Dense(units=num_pixels,input_dim = num_pixels,kernel_initializer='normal',activation='relu'))
    model.add(Dense(units=num_classes,kernel_initializer='normal',activation='softmax'))
    
    # 编译模型
    '''参数选取
    1:损失函数选取：categorical_crossentropy-多类的对数损失
    2:优化器的选取：Adamax，其他还有SGD等
    3:评估标准
    '''
    model.compile(loss='categorical_crossentropy',optimizer='Adamax',metrics=['accuracy'])
    return model

model = create_model()
'''参数含义
1:epochs 指的就是训练过程接中数据将被“轮”多少次”
2:batch_size 就是小批梯度下降算法，把数据分为若干组，称为batch，
按批更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，一批数据中包含的样本数量称为batch_size

'''
model.fit(x_train,y_train,epochs = 3,batch_size = 200)

score = model.evaluate(x_validation,y_validation)

print('MLP: %.2f%%' % (score[1]*100))
    
    

