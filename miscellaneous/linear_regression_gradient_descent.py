import numpy as np
from sklearn import datasets

import matplotlib.pyplot as plt

# setup ground truth coefficientss
param = np.array([1.424921, 2.924613])
bias = 3.189491

# generate origin data
data = np.stack([np.arange(100),np.arange(100)], axis=1)
gt = data @ param.T + bias

# add noise
noise = np.random.normal(size=(100,))
label = gt + noise

def mse(inp, tar):
    return np.mean(np.square(inp-tar))

# initialize parameters
w = np.random.rand(2)
b = np.random.rand()

# settings
lr = 1e-5
epoch = 1000
train_size = 80
test_size = 20

# split data
train_x = data[:train_size]
train_y = label[:train_size]
test_x = data[-test_size:]
test_y = label[-test_size:]

# training loop
for i in range(epoch):
    if i+1 % 400 == 0:
        lr *= .1
    y_hat = train_x @ w.T + b
    loss = mse(y_hat, train_y)
    print(f'Ep.{i+1} - loss: {np.mean(loss)}')
    dw = - 2./train_size * (train_x.T @ (train_y-y_hat).T).T
    db = - 2. * np.mean(train_y-y_hat)
    w = w - lr * dw
    b = b - lr * db        

# test
y_hat = test_x @ w.T + b

# plot
plt.plot(range(test_size), y_hat, 'b')
plt.plot(range(test_size), test_y, 'r')
plt.show()