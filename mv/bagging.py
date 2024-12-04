# Bagging (Bootstrap Aggregating) Algorithm from Scratch
# Bagging (Bootstrap Aggregating) is an ensemble method that combines multiple models (usually the same type, e.g., decision trees) to improve the performance and reduce variance. It works by creating multiple subsets of the original dataset using random sampling with replacement (bootstrap sampling), training a model on each subset, and then averaging (for regression) or voting (for classification) on the predictions of all models.

# Key Concepts of Bagging:
# Bootstrap Sampling: A random subset of the training data is selected with replacement, meaning some data points may appear multiple times, while others may not appear at all.
# Model Training: A model (e.g., decision tree) is trained on each bootstrap sample.
# Aggregation: For classification, the final prediction is made by a majority vote of the individual models. For regression, it is the average of the predictions.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Decision Tree Classifier (from scratch) for Bagging
class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # If all labels are the same, return a leaf node
        if len(set(y)) == 1:
            return {'label': y[0]}

        if depth >= self.max_depth:
            return {'label': np.argmax(np.bincount(y))}

        # Find the best split
        best_split = self._best_split(X, y)
        if best_split is None:
            return {'label': np.argmax(np.bincount(y))}

        left_X, left_y, right_X, right_y = self._split_data(X, y, best_split)
        left_node = self._build_tree(left_X, left_y, depth + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1)

        return {'split': best_split, 'left': left_node, 'right': right_node}

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = None
        n_features = X.shape[1]

        # Loop through all features to find the best split
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask
                left_y = y[left_mask]
                right_y = y[right_mask]
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gini = self._gini_impurity(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)

        return best_split

    def _gini_impurity(self, left_y, right_y):
        total_size = len(left_y) + len(right_y)
        left_size = len(left_y)
        right_size = len(right_y)

        left_gini = 1.0 - sum([(np.sum(left_y == c) / left_size) ** 2 for c in np.unique(left_y)])
        right_gini = 1.0 - sum([(np.sum(right_y == c) / right_size) ** 2 for c in np.unique(right_y)])

        return (left_size / total_size) * left_gini + (right_size / total_size) * right_gini

    def _split_data(self, X, y, split):
        feature_index, threshold = split
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _predict_one(self, x, node):
        if 'label' in node:
            return node['label']
        feature_index, threshold = node['split']
        if x[feature_index] <= threshold:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])


# Bagging Classifier (from scratch)
class BaggingClassifier:
    def __init__(self, base_model, n_estimators=10):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            
            model = self.base_model()
            model.fit(X_bootstrap, y_bootstrap)
            self.models.append(model)

    def predict(self, X):
        # Collect predictions from all models and return the majority vote
        predictions = np.array([model.predict(X) for model in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# Generate synthetic classification dataset
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=0, random_state=42)

# Create a Bagging model using Decision Trees as the base model
bagging = BaggingClassifier(base_model=DecisionTree, n_estimators=50)

# Train the Bagging model
bagging.fit(X, y)

# Predict using the trained model
y_pred = bagging.predict(X)

# Evaluate the accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy of Bagging model: {accuracy * 100:.2f}%")

# Visualizing a subset of predictions (if 2D for simplicity)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
plt.title("Bagging Model Predictions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
