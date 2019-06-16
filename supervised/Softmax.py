import numpy as np
from .SupervisedBase import SupervisedBaseClass

class Softmax(SupervisedBaseClass):

    def __init__(self, lr=0.001):
        self.__lr = lr
        self.n_samples = 0
        self.__w = None

    def _softmax(self, x):
        pred = np.exp(x.dot(self.__w.T))
        return pred/(np.sum(pred, axis=1, keepdims=True)+1e-6)
    
    def _error_calc(self, pred, y):
        nll_loss = -np.log(pred[range(pred.shape[0]),y])
        return nll_loss

    def _update(self, X, y, err):
        diff = np.zeros((self.__n_class,X.shape[1]))
        for i in set(y):
            idx = y==i
            mean_diff = np.mean((X[idx].T*err[idx]).T, axis=0)
            diff[i,:] = mean_diff
        self.__w = self.__w+self.__lr*diff

    def train(self, X, y, batch_size=64, epoch=100):
        X = self._format_batch(X)
        self.__n_class = len(set(y))
        self.__w = np.random.rand(len(set(y)), X.shape[1])
        self._optimize_gd(X, y, batch_size, epoch)

    def _predict(self, X):
        return self._softmax(self._format_batch(X))

    def predict(self, X):
        return np.argmax(self._predict(X), axis=1)
    
    def score(self, X, y, output=True):
        return self._cls_score(X, y, output)