import numpy as np
from .SupervisedBase import SupervisedBaseClass

class Perceptron(SupervisedBaseClass):

    def __init__(self, lr=1e-3):
        self.__lr = lr
        self.__W = None
        self.__bias = None
    
    def __change_label(self, y):
        y[y==0] = -1
        return y

    def __sign(self, arr):
        for i in range(len(arr)):
            if arr[i] < 0:
                arr[i] = -1
            else:
                arr[i] = 1
        return arr

    def _error_calc(self, pred, y):
        diff = pred-y
        err = np.nonzero(diff)[0]
        return 1./len(err), err

    def _update(self, X, y, err):
        for j in err[1]:
            self.__W = self.__W+self.__lr*y[j]*X[j].transpose()
            self.__bias = self.__bias+self.__lr*y[j]

    def train(self, X, y, batch_size=64, epoch=100, method='normal'):
        X, y = self._format_batch(X, y)
        if len(set(y)) > 2:
            raise ValueError("only binary classification supported")
        y = __change_label(y)
        self.__W = np.random.rand(X.shape[1])
        self.__bias = 0
        if method == 'normal':
            self._optimize_gd(X, y, batch_size, epoch)
        elif method == 'dual':
            pass
    
    def _predict(self, X):
        X = self._format_batch(X)
        pred = X.dot(self.__W).flatten()+self.__bias
        return pred

    def predict(self, X):
        return self.__sign(self._predict(X))
