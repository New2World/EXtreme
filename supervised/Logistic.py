import numpy as np
from supervised.SupervisedBase import SupervisedBaseClass as SupervisedBaseClass

class Logistic(SupervisedBaseClass):
    def __init__(self, lr=.01, *args):
        super(Logistic, self).__init__(*args)
        self.__w = None
        self.__b = None
        self.__lr = lr
    
    def __linear(self, x):
        return self.__w.dot(x.T)+self.__b
    
    def __sigmoid(self, x):
        return 1/(1+np.exp(-self.__linear(x)))
    
    def _error_calc(self, pred, y):
        """
        cost(h(x), y) = -log(h(x)) when y == 1, or -log(1-h(x)) when y == 0;
        J = -1/m [sum_{i=1}^m [y_i log{h(x_i)} + (1-y_i) log{1-h(x_i)}]]
        """
        return np.sum(pred-y)
    
    def _update(self, X, y, err):
        dw = np.sum(X*err, axis=0)/X.shape[0]
        db = err/X.shape[0]
        self.__w = self.__w - self.__lr*dw
        self.__b = self.__b - self.__lr*db
    
    def _predict(self, X):
        return self.__sigmoid(self._format_batch(X))
    
    def train(self, X, y, batch_size=32, epoch=10):
        X, y = self._format_batch(X, y)
        self.__w = np.random.rand(X.shape[1])
        self.__b = np.random.rand()
        self._optimize_gd(X, y, batch_size, epoch)
    
    def predict(self, X):
        return self._sign(self._predict(X), thresh=.5).astype(np.int)
    
    def score(self, X, y, output=True):
        return self._cls_score(X, y, output)