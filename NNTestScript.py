import numpy as np
import sklearn.metrics
from sklearn import datasets
from NN import FNN
import seaborn
import matplotlib.pyplot as plt

digit = datasets.load_digits()
idx = np.random.uniform(0, 1500, (1500, )).astype('int')
idx = set(idx)
ridx = set(range(1797))-idx
idx, ridx = np.array(list(idx)), np.array(list(ridx))
X, y, test, label = digit.data[idx,:], digit.target[idx], digit.data[ridx,:], digit.target[ridx]

nn = FNN(layer_nodes=[64, 128, 100, 10], activation='sigmoid')
nn.train(X, y, epoch=5000)
y_hat = nn.predict(test)
confusion_matrix = sklearn.metrics.confusion_matrix(y_hat, label)
plt.title("Confusion Matrix")
ax = seaborn.heatmap(confusion_matrix)
print (1.*len([item for item in zip(y_hat, label) if item[0] == item[1]])/len(label))
plt.show()
