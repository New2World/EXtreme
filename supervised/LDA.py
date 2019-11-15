import numpy as np
from .SupervisedBase import SupervisedBaseClass

class LDA(SupervisedBaseClass):
    def __init__(self, thresh=0.95):
        self.__K = 2
        self.thresh = thresh
        self.__u = None
        self.__solution_space = None

    def __global_scatter(self, X, y):
        St = np.zeros((X.shape[1],X.shape[1]))
        for k in range(self.K):
            whiten = X[y==k] - np.mean(X, axis=0)
            St = St + whiten.T @ whiten
        return St
    
    def __intra_scatter(self, X, y):
        u = np.array([np.mean(X[y==k], axis=0) for k in range(self.K)])
        return u

    def __inter_scatter(self, X, y):
        Sw = np.zeros((X.shape[1],X.shape[1]))
        for k in range(self.K):
            whiten = X[y==k] - self.__u[k]
            Sw = Sw + whiten.T @ whiten
        return Sw
    
    def train(self, X, y):
        self.K = len(set(y))
        X, y = self._format_batch(X, y)
        self.__u = self.__intra_scatter(X, y)
        Sw = self.__inter_scatter(X, y)
        St = self.__global_scatter(X, y)
        Sb = St - Sw

        u, sig, v = np.linalg.svd(Sw)
        Sw_inv = v.T @ np.linalg.inv(np.diag(sig)) @ u
        w = Sw_inv @ Sb

        eigval, eigvec = np.linalg.eig(w)
        order = np.argsort(eigval)[::-1]
        tot_eigval = np.sum(eigval)
        proportion = eigval / tot_eigval
        eigvec = eigvec[:,order]

        m = 0
        tot_proportion = 0.0
        while tot_proportion < self.thresh:
            tot_proportion += proportion[m]
            m += 1
        self.__solution_space = eigvec[:m]
        self.__u = self.__u @ self.__solution_space.T
    
    def _predict(self, X):
        X = self._format_batch(X)
        U = X @ self.__solution_space.T
        pred = []
        for u in U:
            pred.append(np.argmin(np.sum(np.abs(self.__u - u), axis=1)))
        return np.array(pred)

    def predict(self, X):
        return self._predict(X)
    
    def score(self, X, y, output=True):
        return self._cls_score(X, y, output)