import numpy as np
from .SupervisedBase import SupervisedBaseClass

class NaiveBayes(SupervisedBaseClass):
    def __init__(self, eps=1e-6, *args):
        super(NaiveBayes, self).__init__(*args)
        self.__feat_chart = {}
        self.__label_chart = {}
        self.__n_class = 0
        self.__n_feature = 0
        self.__n_sample = 0
        self.__classes = None
        self.__eps = eps
    
    def __count_label(self, y):
        self.__classes = np.unique(y)
        self.__n_sample = len(y)
        self.__n_class = len(self.__classes)
        for c in self.__classes:
            self.__label_chart[c] = np.sum(y==c) / self.__n_sample

    def __count_feat(self, X, y):
        self.__n_feature = X.shape[1]
        for c in self.__classes:
            samples = X[y==c].T
            n_sample = samples.shape[1]
            self.__feat_chart[c] = []
            for f in range(self.__n_feature):
                self.__feat_chart[c].append({})
                values, counts = np.unique(samples[f], return_counts=True)
                for i in range(len(values)):
                    self.__feat_chart[c][f][values[i]] = counts[i] / n_sample

    def train(self, X, y):
        X, y = self._format_batch(X, y)
        self.__count_label(y)
        self.__count_feat(X, y)
    
    def _predict(self, X):
        X = self._format_batch(X)
        result = np.ones((X.shape[0], self.__n_class))
        for i in range(X.shape[0]):
            for c in self.__classes:
                result[i][c] *= self.__label_chart[c]
                for f in range(self.__n_feature):
                    if X[i,f] not in self.__feat_chart[c][f].keys():
                        result[i][c] *= self.__eps
                    else:
                        result[i][c] *= self.__feat_chart[c][f][X[i,f]]
        return result
    
    def predict(self, X):
        return self.__classes[np.argmax(self._predict(X), axis=1)]
    
    def score(self, X, y, output=True):
        return self._cls_score(X, y, output)
    
    def get_chart(self):
        return self.__feat_chart, self.__label_chart