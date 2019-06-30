import numpy as np
import random as rd
import copy as cp

from .SupervisedBase import SupervisedBaseClass

class FNN(SupervisedBaseClass):

    def __init__(self,
                layer_nodes=[2, 3, 2],
                activation='sigmoid',
                reg_lambda=1e-2,
                lr=1e-2,
                weight=None,
                bias=None):
        """
        initialization
        """
        self.__W = []
        self.__b = []
        self.__val = []
        self.__reg_lambda = reg_lambda
        self.__lr = lr
        self.__layers = len(layer_nodes)
        self.__layer_nodes = layer_nodes
        self.__activation = self.__parse_activation(activation)
        self.__build_model(weight, bias)

    def __parse_activation(self, activation):
        """
        get the activation and its corresponding derivation
        """
        func = {'activation':None, 'derivation':None}
        if activation == 'sigmoid':
            func['activation'] = lambda x: 1./(1+np.exp(-x))
            func['derivation'] = lambda x: x*(1-x)
        elif activation == 'tanh':
            func['activation'] = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            func['derivation'] = lambda x: 1-np.power(x, 2)
        elif activation == 'relu':
            func['activation'] = lambda x: np.abs(x*(x>0))
            func['derivation'] = lambda x: np.ones(x.shape)*(x>0)
        return func

    def __build_model(self, weight, bias):
        """
        __W is the weight with shape consist of two interfacing __layers
        __b is the bias
        """
        if weight is None:
            for i in range(self.__layers-1):
                self.__W.append(np.random.randn(self.__layer_nodes[i],
                              self.__layer_nodes[i+1])/np.sqrt(self.__layer_nodes[i]))
        else:
            self.__W = cp.deepcopy(weight)
        if bias is None:
            for i in range(self.__layers-1):
                self.__b.append(np.zeros((1, self.__layer_nodes[i+1])))
        else:
            self.__b = cp.deepcopy(bias)

    def __softmax(self, arr):
        """
        softmax
        """
        exp_scores = np.exp(arr)
        return exp_scores/np.sum(exp_scores, axis=1, keepdims=True)

    def _error_calc(self, pred, y):
        pred[range(pred.shape[0]), y] -= 1
        return pred
    
    def _update(self, X, y, err):
        self.__backpropagation(err)
        self.__val = []

    def __forwardpropagation(self, X):
        """
        X is a sample with shape (batch_size, feature)
        """
        self.__val.append(X)
        for i in range(self.__layers-2):
            z = self.__val[-1].dot(self.__W[i])+self.__b[i]
            self.__val.append(self.__activation['activation'](z))
        output = self.__val[-1].dot(self.__W[self.__layers-2])+self.__b[self.__layers-2]
        return self.__softmax(output)

    def __backpropagation(self, delta):
        """
        derivation of the standard BackPropagation Algorithm
        return error value
        """
        for i in range(self.__layers-2, -1, -1):
            dW = (self.__val[i].T).dot(delta)+self.__reg_lambda*self.__W[i]
            db = np.sum(delta, axis=0, keepdims=True)
            delta = delta.dot(self.__W[i].T)*self.__activation['derivation'](self.__val[i])
            self.__W[i] += -self.__lr*dW
            self.__b[i] += -self.__lr*db

    def train(self, X, y, batch_size=32, epoch=100):
        """
        forward propagation & backpropagation
        """
        X, y = self._format_batch(X, y)
        self.sample, self.feature = X.shape
        self._optimize_gd(X, y, batch_size, epoch)

    def _predict(self, X):
        X = self._format_batch(X)
        return self.__forwardpropagation(X)

    def predict(self, X):
        """
        its kernel is the forward propagation precedure
        """
        output = self._predict(X)
        self.__val = []
        return output

    def score(self, X, y, output=True):
        """
        predict with forward propagation and evaluate the result
        """
        output_prob = self.predict(X)
        output_label = np.argmax(output_prob, axis=1)
        accuracy = 1-(1.0*len(np.nonzero(y-output_label)[0])/len(y))
        print (f"Accuracy: {accuracy}")
