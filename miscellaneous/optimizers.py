import numpy as np

class Optimizer:
    def __init__(self, gradient, lr):
        self.gradient = gradient
        self.lr = lr
        self.grad = 0.0
    
    def _update(self, params, delta):
        return params - delta

    def step(self, params):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, gradient, lr=1e-3, lr_decay=1.0, momentum=0.9, nesterov=True):
        super(SGD, self).__init__(gradient, lr)
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.nesterov = nesterov

    def step(self, params):
        if self.nesterov:
            gradient = self.gradient(params - self.lr*self.grad)
        else:
            gradient = self.gradient(params)
        self.grad = self.momentum * self.grad + (1 - self.momentum) * gradient
        params = self._update(params, self.lr * self.grad)
        self.lr = self.lr * self.lr_decay
        return params

class AdaGrad(Optimizer):
    def __init__(self, gradient, lr=1e-2, epsilon=1e-8):
        super(AdaGrad, self).__init__(gradient, lr)
        self.epsilon = epsilon
    
    def step(self, params):
        gradient = self.gradient(params)
        self.grad = self.grad + gradient * gradient
        delta = self.lr / (np.sqrt(self.grad) + self.epsilon) * gradient
        params = self._update(params, delta)
        return params

class AdaDelta(Optimizer):
    def __init__(self, gradient, lr=1e-3, rho=0.9, epsilon=1e-8):
        super(AdaDelta, self).__init__(gradient, lr)
        self.rho = rho
        self.epsilon = epsilon
        self.update = 0.0
    
    def step(self, params):
        gradient = self.gradient(params)
        self.grad = self.rho * self.grad + (1 - self.rho) * gradient * gradient
        delta = np.sqrt(self.update + self.lr) / np.sqrt(self.grad + self.lr) * gradient
        self.update = self.rho * self.update + (1 - self.rho) * delta
        params = self._update(params, delta)
        return params

class RMSprop(Optimizer):
    def __init__(self, gradient, lr=1e-3, rho=0.9, epsilon=1e-8):
        super(RMSprop, self).__init__(gradient, lr)
        self.rho = rho
        self.epsilon = epsilon
    
    def step(self, params):
        gradient = self.gradient(params)
        self.grad = self.rho * self.grad + (1 - self.rho) * gradient * gradient
        delta = self.lr / (np.sqrt(self.grad) + self.epsilon) * gradient
        params = self._update(params, delta)
        return params

class Adam(Optimizer):
    def __init__(self, gradient, lr=1e-3, rho1=0.9, rho2=0.99, epsilon=1e-8):
        super(Adam, self).__init__(gradient, lr)
        self.rho1 = rho1
        self.rho = rho2
        self.grad1 = 0.0
        self.bias1 = 1.0
        self.bias = 1.0
        self.epsilon = epsilon
    
    def step(self, params):
        gradient = self.gradient(params)
        self.grad1 = self.rho1 * self.grad1 + (1 - self.rho1) * gradient
        self.bias1 = self.bias1 * self.rho1
        grad1 = self.grad1 / (1 - self.bias1)
        self.grad = self.rho * self.grad + (1 - self.rho) * gradient * gradient
        self.bias = self.bias * self.rho
        grad = self.grad / (1 - self.bias)
        delta = self.lr * grad1 / (np.sqrt(grad) + self.epsilon)
        params = self._update(params, delta)
        return params


loss = lambda x: np.sum(x**3-x**2-x)
grad = lambda x: 3*x**2-2*x-1

sgd = SGD(grad, lr=3.4e-3, lr_decay=0.99, momentum=0.0, nesterov=False)
sgdm = SGD(grad, lr=4e-4, lr_decay=0.99, momentum=0.9, nesterov=False)
nsgd = SGD(grad, lr=3.6e-4, lr_decay=1, momentum=0.9, nesterov=True)
adagrad = AdaGrad(grad, lr=98)
adadelta = AdaDelta(grad, lr=960)
rmsprop = RMSprop(grad, lr=31)
adam = Adam(grad, lr=3)

x = np.array([99])
prev = np.zeros((2,))
eps = 1e-6
count = 0

while np.sum(np.abs(x-prev)) > eps:
    prev = x
    x = sgd.step(x)
    count += 1
    if count >= 5000:
        break
    # print(x)

print(f'{count} iterations: x = {x}')