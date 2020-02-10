import numpy as np
from supervised.SupervisedBase import SupervisedBaseClass as SupervisedBaseClass

class Perceptron(SupervisedBaseClass):

    def __init__(self, lr=1e-3):
        self.__w = None
        self.__b = None
        self.__lr = lr

    def _error_calc(self, pred, y):
        diff = pred-y
        err = np.nonzero(diff)[0]
        return 1./len(err), err

    def _update(self, X, y, err):
        for j in err[1]:
            dw = X[j].T
            db = 1.
            if y[j] == 0:
                dw, db = -dw, -db
            self.__w = self.__w+self.__lr*dw
            self.__b = self.__b+self.__lr*db

    def train(self, X, y, batch_size=32, epoch=100, method='normal'):
        X, y = self._format_batch(X, y)
        if len(set(y)) > 2:
            raise ValueError("only binary classification supported")
        self.__w = np.random.rand(X.shape[1])
        self.__b = 0
        if method == 'normal':
            self._optimize_gd(X, y, batch_size, epoch)
        elif method == 'dual':
            pass
    
    def _predict(self, X):
        X = self._format_batch(X)
        pred = X.dot(self.__w).flatten()+self.__b
        return pred

    def predict(self, X):
        return self._sign(self._predict(X))

    def score(self, X, y, output=True):
        return self._cls_score(X, y, output)