import numpy as np
import scipy as sp

class SupervisedBaseClass(object):
  
  def formatData(self, *data):
    """
    convert list to np.ndarray, and get rid of the redundant dimension
    """
    result = []
    for item in data:
      if isinstance(item, list):
        item = np.array(item)
      if isinstance(item, np.ndarray) and item.ndim > 1:
        shape = item.shape
        while shape[0] == 1:
          item.reshape(shape[1:])
          shape = item.shape
      result.append(item)
    return result
  
  def loadData(self, fileName):
    """
    load data from the given file
    """
    X = []; y = []
    with open(fileName, 'r') as dataFile:
      for dataLine in dataFile:
        data = dataLine.split()
        X.append(data[:-1])
        y.append(data[-1])
    return X, y
  
  def _initSize(self, X, y):
    self.sample, self.feature = X.shape
    self.K = len(set(y))
  
  def _errorCalc(self, pred, y):
    pass
  
  def _update(self, X, pred, y, err):
    pass
  
  def _optimize_gd(self, X, y, max_iter, tol):
    error = 0
    for t in xrange(max_iter):
      pred = self.predict(X)
      err = self._errorCalc(pred, y)
      error = err[0]
      self._update(X, pred, y, err)
      if error < tol:  break
  
  def predict(self, X):
    """
    output list
    """
    pass
  
  def score(self, X, y, output=True):
    """
    calculate the accurary of the model
    """
    _y = self.predict(X)
    y = np.array(y); _y = np.array(_y)
    precious = 1-(1.0*len(np.nonzero(y-_y)[0])/len(y))
    if output:
      print "Accurary: %.2f" % precious
    return precious

