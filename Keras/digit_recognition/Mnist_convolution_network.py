
# coding: utf-8

# In[ ]:


'''
卷积神经网络实现手写数字识别
备注：没有添加drop
'''


# In[14]:


from keras.utils import np_utils
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np


# In[11]:


def loadData(path='mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'],f['y_train']
    x_test, y_test = f['x_test'],f['y_test']
    f.close()
    return (x_train,y_train),(x_test,y_test)

# 从Keras导入Mnist数据集
(X_train, y_train), (X_test, y_test) = loadData()


'''
画出一部分图像进行展示
'''
fig = plt.figure()
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(X_train[i],cmap='gray',interpolation='none')
    plt.title("Digit: {}".format(y_train[i]))
    plt.xticks([])
    plt.yticks([])
fig


# In[15]:


'''使用Keras是必须显式声明输入图像深度的尺寸。例如，具有所有3个RGB通道的全色图像的深度为3。
MNIST图像的深度为1，但我们必须明确声明,也就是说，我们希望将数据集从形状(n,rows,cols)转换为(n,rows,cols,channels)。
'''

# 输入图像尺寸
img_x,img_y = 28,28

X_train =  X_train.reshape(X_train.shape[0],img_x,img_y,1).astype('float') 
X_test = X_test.reshape(X_test.shape[0],img_x,img_y,1).astype('float') 

# 数据标准化
X_train = X_train /255
X_test = X_test /255


#由于最终的输出结果是0-10的数字，属于多目标预测，因此进行one_hot编码，提高效率
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test) # y_test 由一维矩阵转换为二维矩阵


# 定义模型结构
model = Sequential()
model.add(Conv2D(32,kernel_size=(5,5),activation='relu',input_shape=(img_x,img_y,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

# 编译
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#训练
model.fit(X_train, y_train, batch_size=200, epochs=3)


score = model.evaluate(X_test,y_test)

print('MLP: %.2f%%' % (score[1]*100))


# In[17]:




