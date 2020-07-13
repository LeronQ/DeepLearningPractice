
# coding: utf-8

# In[21]:


'''
    梯度下降-所有样本求损失
'''

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x*x 

def cost(xs,ys):
    cost = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred-y)**2
    return cost/len(xs)

def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2*x*(x*w-y)
    return grad/len(xs)

loss = []
for epoch in range(1000):
    cost_val = cost(x_data,y_data)
    loss.append(cost_val)
    grad_val = gradient(x_data,y_data)
    w -= 0.01 * grad_val
    print("Epoch:",epoch,"w:",w,"loss:",cost_val)

print("predict after training",4,forward(4))
        
        
    


# In[23]:


'''
    随机梯度下降-每个样本求损失
'''

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x*x 

def loss(xs,ys):
    y_pred = forward(x)
    return (y_pred-y)**2

def gradient(x,y):
    return 2*x*(x*w-y)

for epoch in range(1000):
    for x,y in zip(x_data,y_data):
        grad = gradient(x,y)
        w -= 0.01*grad
        print("\tgrad",x,y,grad)
        l = loss(x,y)
        
    print("Epoch:",epoch,"w:",w,"loss:",l)


        
    

