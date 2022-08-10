from sklearn.tree import DecisionTreeClassifier


class Node:
    def __init__(self,
                 feature_name,
                 feature_set,
                 criterion,
                 is_leaf=False):
        self.left = None
        self.right = None

    def _find_split(self):
        pass

    def _check_stop(self):
        pass


class Tree:
    def __init__(self,
                 criterion,
                 min_split_samples=5,
                 min_leff_samples=5,
                 min_cri=0.1,
                 ):
        self.criterion = criterion
        self.min_split_samples = min_split_samples
        self.min_leff_samples = min_leff_samples
        self.min_cri = min_cri
        self.root = None

    def fit(self):
        pass

    def predict(self):
        pass