#!/bin/env python

#from Preprocess import *
from PLA import Perceptron
from random import uniform
import matplotlib.pyplot as plt

def draw(X, W, b):
  x1 = [x[0] for x in X[:45]]; y1 = [x[1] for x in X[:45]]
  x2 = [x[0] for x in X[46:]]; y2 = [x[1] for x in X[46:]]
  plt.scatter(x1, y1, s=20, c='green')
  plt.scatter(x2, y2, s=20, c='blue')
  x3 = range(-5, 25); y3 = [-(W[0]*x+b)/W[1] for x in x3]
  y4 = [1.2*x+1 for x in x3]
  # red line: the line model learnt
  plt.plot(x3, y3, 'r-')
  # yellow line: the target function
  plt.plot(x3, y4, 'y-')
  plt.show()

if __name__ == '__main__':
  X = []
  y = [1]*45+[-1]*55
  for label in y:
    x1 = uniform(0, 20)
    x2 = 1.2*x1+1+label*uniform(0, 5)
    x = [x1, x2]
    X.append(x)
  
  mod = Perceptron()
  mod.train(X, y)
  mod.score(X, y)
  draw(X, mod.W, mod.bias)
