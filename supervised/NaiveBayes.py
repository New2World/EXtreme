import numpy as np
from .SupervisedBase import SupervisedBaseClass

class NaiveBayes(SupervisedBaseClass):
    """
    TOO SLOW!!!!!
    """
    def __init__(self, *args):
        super(NaiveBayes, self).__init__(*args)
        self.__n_class = 0
        self.__y_prob = None
        self.__X_prob = None
    
    def __label_prob(self, y):
        uniq_y, y_count = np.unique(y, return_counts=True)
        self.__n_class = uniq_y.shape[0]
        self.__y_prob = y_count / float(y.shape[0])
    
    def __get_prob(self, feature, val, label):
        all_sample = np.arange(self.__n_sample)
        X_with_label = self.__train_data[self.__train_label==label]
        denominator = 0.
        if X_with_label is not None:
            denominator = X_with_label.shape[0]
        X_with_feature_label = X_with_label[X_with_label[:,feature]==val]
        numerator = 0.
        if X_with_feature_label is not None:
            numerator = X_with_feature_label.shape[0]
        return (numerator+1.) / (denominator+self.__n_class)
    
    def train(self, X, y):
        X, y = self._format_batch(X, y)
        self.__n_sample, self.__n_feature = X.shape
        self.__label_prob(y)
        self.__train_data = X
        self.__train_label = y
    
    def _predict(self, X):
        X = self._format_batch(X)
        result = np.ones((X.shape[0], self.__n_class))
        for sample in range(X.shape[0]):
            for label in range(self.__n_class):
                for feature in range(self.__n_feature):
                    result[sample,label] *= self.__get_prob(feature, X[sample,feature], label)
        return result
    
    def predict(self, X):
        return np.argmax(self._predict(X), axis=1)
    
    def score(self, X, y, output=True):
        return self._cls_score(X, y, output)