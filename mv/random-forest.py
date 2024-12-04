import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gini Index calculation
def gini_index(y):
    class_probs = [np.sum(y == c) / len(y) for c in np.unique(y)]
    return 1 - sum(p ** 2 for p in class_probs)

# Split dataset based on a feature and a threshold
def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# Find the best split for a dataset
def best_split(X, y):
    best_gini = float("inf")
    best_split = None
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
            gini_left = gini_index(y_left)
            gini_right = gini_index(y_right)
            gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right
            if gini < best_gini:
                best_gini = gini
                best_split = (feature_index, threshold)
    return best_split

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._fit(X, y, depth=0)

    def _fit(self, X, y, depth):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return DecisionTreeNode(value=np.argmax(np.bincount(y.flatten())))

        feature_index, threshold = best_split(X, y)
        X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)

        left_node = self._fit(X_left, y_left, depth + 1)
        right_node = self._fit(X_right, y_right, depth + 1)

        return DecisionTreeNode(feature_index, threshold, left_node, right_node)

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

# Bootstrap Sampling: Create random subsets of the original dataset
def bootstrap_sample(X, y):
    indices = np.random.choice(len(X), size=len(X), replace=True)
    return X[indices], y[indices]

# Random Forest Classifier
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt"):
        self.n_estimators = n_estimators  # Number of trees
        self.max_depth = max_depth  # Max depth of each tree
        self.max_features = max_features  # Max number of features to consider for each split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Create a bootstrap sample of the data
            X_sample, y_sample = bootstrap_sample(X, y)

            # If max_features is "sqrt", choose sqrt(n_features) random features
            if self.max_features == "sqrt":
                max_features = int(np.sqrt(X.shape[1]))
            else:
                max_features = X.shape[1]
            
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Get predictions from all trees and aggregate by majority voting
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        majority_preds = [np.bincount(tree_pred).argmax() for tree_pred in tree_preds.T]
        return np.array(majority_preds)

# Load dataset (using sklearn for easy testing)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data[:, :2]  # Use only first two features for easy visualization
y = iris.target

# Train the random forest
rf = RandomForest(n_estimators=10, max_depth=5)
rf.fit(X, y)

# Predict using the random forest
y_pred = rf.predict(X)

# Calculate accuracy
accuracy = np.mean(y_pred == y)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualize decision boundary (for 2D data)
X_test = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
Y_test = np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
xx, yy = np.meshgrid(X_test, Y_test)
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.title("Random Forest: Actual vs Predicted (Iris Dataset)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

