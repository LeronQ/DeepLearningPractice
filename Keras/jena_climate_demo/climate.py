
# coding: utf-8

# In[1]:


#观察数据集中数据
import os
# data_dir='D:\\jupyter_code\\jena_climate'
data_dir='C:\\Users\\10189\\Desktop\\jupyter\\Deep_learning\\jena_climate'
# C:\Users\10189\Desktop\jupyter\Deep_learning\jena_climate
fname=os.path.join(data_dir,'jena_climate_2009_2016.csv')
 
f=open(fname)
data=f.read()
f.close()
 
lines=data.split('\n')
header=lines[0].split(',')
lines=lines[1:]
 
print(header)
print(len(lines))
 
import numpy as np
float_data=np.zeros((len(lines),len(header)-1))
for i, line in enumerate(lines):
    values=[float(x) for x in line.split(',')[1:]]
    float_data[i,:]=values
 
from matplotlib import pyplot as plt
temp=float_data[:,1]
plt.plot(range(len(temp)),temp)
 
plt.plot(range(1440),temp[:1440])
 
#数据标准化（减去平均数，除以标准差）
# 前200000个数据作为训练数据，所以只对这部分数据计算平均值和标准差
mean=float_data[:200000].mean(axis=0)
float_data-=mean
std=float_data[:200000].std(axis=0)
float_data/=std
 
 
#生成时间序列样本及其目标的生成器
#data:浮点数数据组成的原始数组
#lookback：输入数据应该包括过去多少个时间步
#delay：目标应该在未来多少个时间步后
#min_index,max_index：数组中的索引，界定需要抽取哪些时间步，有助于保存数据用于验证和测试
#shuffle：是否打乱样本
#batch_size：每个批量的样本数
#step：数据采样周期，每个时间步是10min，设置为6，即每小时取一个数据点
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)
        
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices=range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][1]
        yield samples,targets
 
 
#准备训练生成器，验证生成器，测试生成器
#输入数据包括过去10天内的数据，每小时抽取一次数据点
#目标为一天以后的天气，批次样本数为128
lookback=1440 # 过去10天的数据
step=6    # 每一个小时抽取一个数据点
delay=144 #每10分钟为1条记录，所以1天共有144个数据，目标是预测未来24小时的数据
batch_size=128

 
train_gen=generator(float_data,lookback=lookback,delay=delay,min_index=0,max_index=200000,
                   shuffle=True,step=step,batch_size=batch_size)
val_gen=generator(float_data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,
                   shuffle=True,step=step,batch_size=batch_size)
train_gen=generator(float_data,lookback=lookback,delay=delay,min_index=300001,max_index=None,
                   shuffle=True,step=step,batch_size=batch_size)
val_steps=(300000-200001-lookback)//batch_size
test_steps=(len(float_data)-300001-lookback)//batch_size
 


'''generator 参数介绍：
fit_generator(self, generator, steps_per_epoch, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)


（1)返回结果：形如（inputs，targets）或者（inputs, targets,sample_weight）的tuple元组，所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。
每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
（2）steps_per_epoch： 整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
（3）epochs：整数，数据迭代的轮数
（4）verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
（5）validation_data：生成验证集的生成器，一个形如（inputs,targets）的tuple，一个形如（inputs,targets，sample_weights）的tuple
（6）validation_steps: 当validation_data为生成器时，本参数指定验证集的生成器返回次数
（7）workers：最大进程数
（8）max_q_size：生成器队列的最大容量
（9）pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递non 
（10）picklable（无法被pickle序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。
（11）initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用

'''


#训练评估一个密集连接模型
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
 
model=Sequential()
model.add(layers.Flatten(input_shape=(lookback//step,float_data.shape[-1])))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=5,validation_data=val_gen,validation_steps=val_steps)


# In[3]:


from matplotlib import pyplot as plt

loss = history.history['loss']
val_loss =history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Training val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


#训练并评估一个基于GRU的模型
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
 
model=Sequential()
model.add(layers.GRU(32,input_shape=(None,float_data.shape[-1])))
model.add(layers.Dense(1))
 
model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=5,validation_data=val_gen,validation_steps=val_steps)


loss = history.history['loss']
val_loss =history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Training val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:


#训练并评估一个使用dropout正则化的基于GRU的模型
#不再过拟合，分数更稳定
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
 
model=Sequential()
model.add(layers.GRU(32,dropout=0.2,recurrent_dropout=0.2,input_shape=(None,float_data.shape[-1])))
model.add(layers.Dense(1))
 
model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=5,validation_data=val_gen,validation_steps=val_steps)
 
 
 # 绘制结果
from matplotlib import pyplot as plt

loss = history.history['loss']
val_loss =history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Training val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
 
 


# In[ ]:


#训练并评估一个使用dropout正则化的堆叠GRU模型
#有所改进但效果并不明显
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
 
model=Sequential()
model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5,return_sequences=True,input_shape=(None,float_data.shape[-1])))
model.add(layers.GRU(64,activation='rule',dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))
 
model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=5,validation_data=val_gen,validation_steps=val_steps)



# 绘制结果
from matplotlib import pyplot as plt

loss = history.history['loss']
val_loss =history.history['val_loss']

epochs = range(1,len(loss)+1)

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Training val_loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

