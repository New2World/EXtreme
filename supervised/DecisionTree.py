import math
import numpy as np
from supervised.SupervisedBase import SupervisedBaseClass as SupervisedBaseClass

class _tree_node(object):
    def __init__(self, feat_id, label):
        self.id = -1
        self.feat_id = feat_id
        self.label = label
        self.lnode = None
        self.rnode = None

class DecisionTree(SupervisedBaseClass):
    def __init__(self, heuristic='entropy'):
        self.root = None
        self.node_count = 0
        self.n_features = 0
        self.feature_name = None
        self.__heuristic = self.__info_gain if heuristic=='entropy' else self.__impurity
        super(DecisionTree, self).__init__()

    def __entropy(self, y):
        if len(y) == 0:
            return 0
        prob_true = 1.*np.sum(y==1)/len(y)
        prob_false = 1.*np.sum(y==0)/len(y)
        ent_true = 0 if prob_true == 0 else prob_true*math.log(prob_true,2)
        ent_false = 0 if prob_false == 0 else prob_false*math.log(prob_false,2)
        return -(ent_true+ent_false)

    def __cond_ent(self, feat_id, x, y):
        if len(y) == 0:
            return 0
        feat_true = y[x[:,feat_id]==1]
        feat_false = y[x[:,feat_id]==0]
        ent_true = self.__entropy(feat_true)
        ent_false = self.__entropy(feat_false)
        prob_true = 1.*len(feat_true)/len(y)
        prob_false = 1.*len(feat_false)/len(y)
        return prob_true*ent_true+prob_false*ent_false

    def __info_gain(self, feat_id, x, y):
        info_gain = self.__entropy(y)-self.__cond_ent(feat_id, x, y)
        return info_gain

    def __variance_impurity(self, y):
        if len(y) == 0:
            return 0
        prob_true = 1.*np.sum(y==1)/len(y)
        prob_false = 1.*np.sum(y==0)/len(y)
        return prob_true*prob_false

    def __cond_vi(self, feat_id, x, y):
        if len(y) == 0:
            return 0
        feat_true = y[x[:,feat_id]==1]
        feat_false = y[x[:,feat_id]==0]
        vi_true = self.__variance_impurity(feat_true)
        vi_false = self.__variance_impurity(feat_false)
        prob_true = 1.*len(feat_true)/len(y)
        prob_false = 1.*len(feat_false)/len(y)
        return prob_true*vi_true+prob_false*vi_false

    def __impurity(self, feat_id, x, y):
        purity_gain = self.__variance_impurity(y)-self.__cond_vi(feat_id, x, y)
        return purity_gain

    def __build_tree(self, x, y):
        info_gain = [self.__heuristic(i, x, y) for i in range(self.n_features)]
        feat_id = np.argmax(info_gain)
        if info_gain[feat_id] <= 0:
            root = _tree_node(-1, int(round(1.*np.sum(y)/len(y))))
            return root
        x_true, y_true = x[x[:,feat_id]==1], y[x[:,feat_id]==1]
        x_false, y_false = x[x[:,feat_id]==0], y[x[:,feat_id]==0]
        root = _tree_node(feat_id, int(round(1.*np.sum(y)/len(y))))
        root.lnode = self.__build_tree(x_true, y_true)
        root.id = self.node_count
        self.node_count += 1
        root.rnode = self.__build_tree(x_false, y_false)
        return root

    def train(self, X, y, feature_name=None):
        self.root = None
        X, y = self._format_batch(X, y)
        self.n_features = X.shape[1]
        self.feature_name = feature_name
        self.root = self.__build_tree(X, y)
        return self.root

    def _predict(self, X):
        X = self._format_batch(X)
        y = []
        for sample in X:
            ptr = self.root
            while ptr.feat_id >= 0:
                if sample[ptr.feat_id] == 1:
                    ptr = ptr.lnode
                else:
                    ptr = ptr.rnode
            y.append(ptr.label)
        return np.array(y)
    
    def predict(self, X):
        return self._predict(X)
    
    def score(self, X, y):
        return self._cls_score(X, y)

    def __describe_tree(self, root, outp):
        description = ""
        if root.feat_id < 0:
            description += " : {}".format(root.label)
            return description
        if self.feature_name is not None:
            description += "\n"+outp+"{} = 0".format(self.feature_name[root.feat_id])
        else:
            description += "\n"+outp+" 0"
        description += self.__describe_tree(root.rnode, outp+"| ")
        if self.feature_name is not None:
            description += "\n"+outp+"{} = 1".format(self.feature_name[root.feat_id])
        else:
            description += "\n"+outp+" 1"
        description += self.__describe_tree(root.lnode, outp+"| ")
        return description

    def plot_tree(self):
        print(self.__describe_tree(self.root, ""))