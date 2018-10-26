import numpy as np
from SupervisedBase import SupervisedBaseClass

class LinReg(SupervisedBaseClass):

    def __init__(self):
        pass

    def train(self, X, y):
        X, y = self._format_batch(X, y)
        temp_mat = X.transpose().dot(X)
        inv_mat = np.linalg.pinv(temp_mat)
        self.__w = (inv_mat.dot(X.transpose().dot(y.transpose()))).transpose()

    def predict(self, X):
        X = self._format_batch(X)
        return (self.__w.dot(X.transpose())).tolist()
