import numpy as np
from sklearn import datasets
from NN import FNN

iris = datasets.load_digits()
idx = np.random.uniform(0, 1500, (1500, )).astype('int')
idx = set(idx)
ridx = set(range(1797))-idx
idx, ridx = np.array(list(idx)), np.array(list(ridx))
X, y, test, label = iris.data[idx,:], iris.target[idx], iris.data[ridx,:], iris.target[ridx]

nn = FNN(layer_nodes=[64, 128, 100, 10], activation='sigmoid')
nn.train(X, y, epoch=5000)
y_hat = nn.predict(test)
print zip(y_hat, label)
print 1.*len([item for item in zip(y_hat, label) if item[0] == item[1]])/len(label)
