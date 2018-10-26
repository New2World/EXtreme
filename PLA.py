import numpy as np
from SupervisedBase import SupervisedBaseClass

class Perceptron(SupervisedBaseClass):

    def __init__(self, lr=1e-3, tol=1e-6):
        self.alpha = lr
        self.tol = tol

    def __sign(self, arr):
        for i in xrange(len(arr)):
            if arr[i] < 0:
                arr[i] = -1
            else:
                arr[i] = 1
        return arr

    def _error_calc(self, pred, y):
        diff = pred-y
        err = np.nonzero(diff)[0]
        return 1.0*len(err)/self.sample, err

    def _update(self, X, pred, y, err):
        for j in err[1]:
            self.W = self.W+self.alpha*y[j]*X[j].transpose()
            self.bias = self.bias+self.alpha*y[j]

    def train(self, X, y, max_iter=1000, method='normal'):
        X, y = self._format_batch(X, y)
        self.sample, self.feature = X.shape
        self.W = np.random.rand(self.feature)
        self.bias = 0
        if method == 'normal':
            self._optimize_gd(X, y, max_iter, self.tol)
        elif method == 'dual':
            pass

    def predict(self, X):
        X = self._format_batch(X)
        pred = X.dot(self.W).flatten()+self.bias
        return self.__sign(pred)
