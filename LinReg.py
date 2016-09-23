import numpy as np
from base import *

class LinReg(base):
  
  def train(self, X, y):
    X, y = self.formatData(X, y)
    temp_mat = X.transpose().dot(X)
    inv_mat = np.linalg.pinv(temp_mat)
    self.__w = (inv_mat.dot(X.transpose().dot(y.transpose()))).transpose()
  
  def predict(self, X):
    X = self.formatData(X)[0]
    return (self.__w.dot(X.transpose())).tolist()
