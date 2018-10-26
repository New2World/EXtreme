import numpy as np
from SupervisedBase import SupervisedBaseClass

class Softmax(SupervisedBaseClass):

    def __init__(self, alpha=0.001):
        self.alpha = alpha
        self.sample = 0
        self.feature = 0

    def __init_arg(self, X, y):
        self.sample, self.feature = X.shape
        self.K = len(set(y))
        self.theta = np.zeros((self.K, X.shape[1]))

    def __calc_each(self, x):
        pred = np.exp(self.theta*x)
        normalizer = np.sum(pred)
        pred = pred/normalizer
        return pred

    def train(self, X, y, max_iter=1000):
        X = self._format_batch(X)
        self.__init_arg(X, y)
        X = X.T
        loop = 0
        print 'training...'
        while loop < max_iter:
            loop += 1
            for i in xrange(self.sample):
                pred = self.__calc_each(X[:,i])
                updateItem = self.alpha*(1-pred[y[i],0])*X[:,i].T
                self.theta[y[i],:] = self.theta[y[i],:]+updateItem.reshape((1,-1))

    def predict(self, X):
        X = self._format_batch(X)
        X = X.T; result = []
        for i in xrange(X.shape[1]):
            pred = self.__calc_each(X[:,i])
            result.append(np.argmax(pred))
        return result
