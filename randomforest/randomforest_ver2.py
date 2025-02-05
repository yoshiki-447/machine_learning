import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# データの生成
np.random.seed(42)
X = np.random.randn(200, 2)
y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_index = int(X.shape[0] * (1 - test_size))
    train_indices, test_indices = indices[:split_index], indices[split_index:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class DecisionTree:
    def __init__(self, max_depth=None, min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size
        self.tree = None

    def fit(self, X, y):
        dataset = np.hstack((X, y.reshape(-1, 1)))
        self.tree = self._split(dataset)
        
    def predict(self, X):
        predictions = [self._predict_row(row, self.tree) for row in X]
        return np.array(predictions)
    
    def _split(self, dataset, depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        if len(set(y)) == 1 or len(y) <= self.min_size or (self.max_depth and depth >= self.max_depth):
            return Counter(y).most_common(1)[0][0]
        
        best_index, best_value, best_score, best_splits = self._get_best_split(dataset)
        if best_score == float('inf'):
            return Counter(y).most_common(1)[0][0]
        
        left_tree = self._split(best_splits[0], depth + 1)
        right_tree = self._split(best_splits[1], depth + 1)
        return (best_index, best_value, left_tree, right_tree)
    
    def _get_best_split(self, dataset):
        X, y = dataset[:, :-1], dataset[:, -1]
        best_index, best_value, best_score, best_splits = None, None, float('inf'), None
        for index in range(X.shape[1]):
            for value in np.unique(X[:, index]):
                splits = self._test_split(index, value, dataset)
                score = self._gini_index(splits, y)
                if score < best_score:
                    best_index, best_value, best_score, best_splits = index, value, score, splits
        return best_index, best_value, best_score, best_splits
    
    def _test_split(self, index, value, dataset):
        left = dataset[dataset[:, index] < value]
        right = dataset[dataset[:, index] >= value]
        return left, right
    
    def _gini_index(self, splits, classes):
        n_instances = float(sum([len(split) for split in splits]))
        gini = 0.0
        for split in splits:
            size = len(split)
            if size == 0:
                continue
            score = 0.0
            proportions = [np.sum(split[:, -1] == c) / size for c in np.unique(classes)]
            score = sum([p * p for p in proportions])
            gini += (1.0 - score) * (size / n_instances)
        return gini

    def _predict_row(self, row, node):
        if isinstance(node, (int, float)):
            return node
        index, value, left_tree, right_tree = node
        if row[index] < value:
            return self._predict_row(row, left_tree)
        else:
            return self._predict_row(row, right_tree)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_size=1, sample_size=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            sample_X, sample_y = self._bootstrap_sample(X, y)
            tree = DecisionTree(max_depth=self.max_depth, min_size=self.min_size)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        majority_votes = [Counter(row).most_common(1)[0][0] for row in predictions.T]
        return np.array(majority_votes)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

forest = RandomForest(n_trees=10, max_depth=5, min_size=2)
forest.fit(X_train, y_train)
predictions = forest.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='bwr')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap='bwr')
    plt.title("Random Forest Decision Boundary")
    plt.savefig('figure.jpg')

plot_decision_boundary(X_test, y_test, forest)
