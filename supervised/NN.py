import numpy as np
import random as rd
import copy as cp

class FNN():

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

    def __parse_activation(self, __activation):
        """
        get the __activation and its corresponding derivation
        """
        func = {'__activation':None, 'derivation':None}
        if __activation == 'sigmoid':
            func['__activation'] = lambda x: 1./(1+np.exp(-x))
            func['derivation'] = lambda x: x*(1-x)
        elif __activation == 'tanh':
            func['__activation'] = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            func['derivation'] = lambda x: 1-np.power(x, 2)
        elif __activation == 'relu':
            func['__activation'] = lambda x: max(0,x)
            func['derivation'] = lambda x: max(0,x/abs(x))
        return func

    def __get_batch(self, X, y, batch_size):
        """
        choose batch data randomly, with some repeat items
        """
        rand_idx = np.random.uniform(0, self.sample, (batch_size, ))
        rand_idx = rand_idx.astype('int')
        return X[rand_idx,:], y[rand_idx]

    def __format_data(self, *data):
        """
        turn list into np.ndarray
        """
        after = []
        for item in data:
            if isinstance(item, list):
                after.append(np.array(item))
            else:
                after.append(item)
        if len(data) == 1:
            return after[0]
        return after

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

    def __calc_error(self, delta, y):
        log_prob = -np.log(delta[range(delta.shape[0]), y])
        loss = np.sum(log_prob)
        regularizer_loss = sum(map(lambda x: np.sum(np.square(x)), self.__W))
        loss += self.__reg_lambda*regularizer_loss/2
        return loss/len(y)

    def __forwardpropagation(self, X):
        """
        X is a sample with shape (batch_size, feature)
        """
        self.__val.append(X)
        for i in range(self.__layers-2):
            z = self.__val[-1].dot(self.__W[i])+self.__b[i]
            self.__val.append(self.__activation['__activation'](z))
        return self.__val[-1].dot(self.__W[self.__layers-2])+self.__b[self.__layers-2]

    def __backpropagation(self, z, y, batch_size):
        """
        derivation of the standard BackPropagation Algorithm
        return error value
        """
        delta = self.__softmax(z)
        error = self.__calc_error(delta, y)
        delta[range(batch_size), y] -= 1
        for i in range(self.__layers-2, -1, -1):
            dW = (self.__val[i].T).dot(delta)+self.__reg_lambda*self.__W[i]
            db = np.sum(delta, axis=0, keepdims=True)+self.__reg_lambda*self.__b[i]
            delta = delta.dot(self.__W[i].T)*self.__activation['derivation'](self.__val[i])
            self.__W[i] += -self.__lr*dW
            self.__b[i] += -self.__lr*db
        return error

    def train(self, X, y, batch_size=32, epoch=100, show_procedure=True):
        """
        forward propagation & backpropagation
        """
        X, y = self.__format_data(X, y)
        self.sample, self.feature = X.shape

        print ('Start training...')

        for i in range(1, epoch+1):
            batch_train, batch_label = self.__get_batch(X, y, batch_size)
            output_prob = self.__forwardpropagation(batch_train)
            error = self.__backpropagation(output_prob, batch_label, batch_size)
            if show_procedure:
                print (f"Epoch #{i}: loss: {error}")
            self.__val = []

    def predict(self, X):
        """
        its kernel is the forward propagation precedure
        """
        X = self.__format_data(X)
        output = self.__forwardpropagation(X)
        output_prob = self.__softmax(output)
        output_label = np.argmax(output_prob, axis=1)
        self.__val = []
        return output_label

    def eval(self, X, y):
        """
        predict with forward propagation and evaluate the result
        """
        output_label = self.predict(X)
        accuracy = 1.*len([item for item in zip(output_label, y) if item[0] == item[1]])/len(y)
        print (f"Accuracy: {accuracy}")
