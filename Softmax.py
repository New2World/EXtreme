#!/bin/env python

import numpy as np
from SupervisedBase import SupervisedBaseClass

class Softmax(SupervisedBaseClass):

    def __init__(self, alpha=0.001, max_iter=500):
        self.alpha = alpha
        self.max_iter = max_iter

    def __initArg(self, X, y):
        self._initSize(X, y)
        self.theta = np.mat(np.zeros((self.K, X.shape[1])))

    def __calcEach(self, x):
        pred = np.exp(self.theta*x)
        normalizer = np.sum(pred)
        pred = pred/normalizer
        return pred

    def train(self, X, y):
        X = self.formatData(X)[0]
        self.__initArg(X, y)
        X = X.T
        loop = 0
        print 'training...'
        while loop < self.max_iter:
            loop += 1
            for i in xrange(self.sample):
                pred = self.__calcEach(X[:,i])
                updateItem = self.alpha*(1-pred[y[i],0])*X[:,i].T
                self.theta[y[i],:] = self.theta[y[i],:]+updateItem

    def predict(self, X):
        X = self.formatData(X)[0]
        X = X.T; result = []
        for i in xrange(X.shape[1]):
            pred = self.__calcEach(X[:,i])
            result.append(np.argmax(pred))
        return result
