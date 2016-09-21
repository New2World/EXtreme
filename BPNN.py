import numpy as np
import random as rd
import copy as cp

class FNN(object):
  
  W = []
  b = []
  val = []
  
  def __init__(self, layer_nodes=[2, 3, 2], activation='sigmoid', reg_lambda=1e-2, eta=1e-2, weight=None, bias=None):
    """
    initialization
    """
    self.reg_lambda = reg_lambda
    self.eta = eta
    self.layers = len(layer_nodes)
    self.layer_nodes = layer_nodes
    self.activation = self.parse_activation(activation)
    self.build_model(weight, bias)
  
  def parse_activation(self, activation):
    """
    get the activation and its corresponding derivation
    """
    func = {'activation':None, 'derivation':None}
    if activation == 'sigmoid':
      func['activation'] = lambda x: 1./(1+np.exp(-x))
      func['derivation'] = lambda x: x*(1-x)
    elif activation == 'tanh':
      func['activation'] = lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
#      func['activation'] = np.tanh
      func['derivation'] = lambda x: 1-np.power(x, 2)
    return func
  
  def get_batch(self, X, y, batch_size):
    """
    choose batch data randomly, with some repeat items
    """
    idx = np.random.uniform(0, self.sample, (batch_size, ))
    idx = idx.astype('int')
    return X[idx,:], y[idx]
  
  def format_data(self, *data):
    """
    turn list into np.ndarray
    """
    after = []
    for item in data:
      if isinstance(item, list):  after.append(np.array(item))
      else:  after.append(item)
    if len(data) == 1:  return after[0]
    return after
  
  def build_model(self, weight, bias):
    """
    W is the weight with shape consist of two interfacing layers
    b is the bias
    """
    if weight is None:
      for i in xrange(self.layers-1):
        self.W.append(np.random.randn(self.layer_nodes[i], self.layer_nodes[i+1])/np.sqrt(self.layer_nodes[i]))
    else:
      self.W = cp.deepcopy(weight)
    if bias is None:
      for i in xrange(self.layers-1):
        self.b.append(np.zeros((1, self.layer_nodes[i+1])))
    else:
      self.b = cp.deepcopy(bias)
  
  def softmax(self, arr):
    """
    softmax
    """
    exp_scores = np.exp(arr)
    probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    return probs
  
  def forwardpropagation(self, X):
    """
    X is a sample with shape (batch_size, feature)
    """
    self.val.append(X)
    for i in xrange(self.layers-2):
      z = self.val[-1].dot(self.W[i])+self.b[i]
      self.val.append(self.activation['activation'](z))
    return self.val[-1].dot(self.W[self.layers-2])+self.b[self.layers-2]
  
  def backpropagation(self, z, y, batch_size):
    """
    derivation of the standard BackPropagation Algorithm
    """
    delta = self.softmax(z)
    delta[range(batch_size), y] -= 1	# NICE CODE
    for i in xrange(self.layers-2, -1, -1):
      dW = (self.val[i].T).dot(delta)+self.reg_lambda*self.W[i]
      db = np.sum(delta, axis=0, keepdims=True)+self.reg_lambda*self.b[i]
      delta = delta.dot(self.W[i].T)*self.activation['derivation'](self.val[i])
      self.W[i] += -self.eta*dW
      self.b[i] += -self.eta*db
  
  def train(self, X, y, batch_size=32, epoch=100):
    """
    forward propagation & backpropagation
    """
    X, y = self.format_data(X, y)
    self.sample, self.feature = X.shape
    for i in xrange(epoch):
      batch_train, batch_label = self.get_batch(X, y, batch_size)
      output_prob = self.forwardpropagation(batch_train)
      self.backpropagation(output_prob, batch_label, batch_size)
#      output_prob = self.forwardpropagation(X)
#      self.backpropagation(output_prob, y, batch_size)
      self.val = []
  
  def predict(self, X):
    """
    its kernel is the forward propagation precedure
    """
    X = self.format_data(X)
    output = self.forwardpropagation(X)
    output_prob = self.softmax(output)
    output_label = np.argmax(output_prob, axis=1)
    return output_label
