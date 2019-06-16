import numpy as np
from .SupervisedBase import SupervisedBaseClass

class LinReg(SupervisedBaseClass):

    def __init__(self):
        self.__w = None

    def __add_bias(self, X):
        samples = X.shape[0]
        bias = np.ones((samples,1))
        return np.hstack((X, bias))

    def train(self, X, y):
        X, y = self._format_batch(X, y)
        X = self.__add_bias(X)
        temp_mat = X.transpose().dot(X)
        inv_mat = np.linalg.pinv(temp_mat)
        self.__w = (inv_mat.dot(X.transpose().dot(y.transpose()))).transpose()

    def _predict(self, X):
        X = self._format_batch(X)
        X = self.__add_bias(X)
        return self.__w.dot(X.transpose())
    
    def predict(self, X):
        return self._predict(X)
    
    def score(self, X, y, output=True):
        return self._reg_score(X, y, output)
