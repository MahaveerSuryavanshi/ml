import numpy as np
import matplotlib.pyplot as plt

# Gini Index calculation
def gini_index(y):
    """
    Calculate the Gini index for a set of labels.
    """
    class_probs = [np.sum(y == c) / len(y) for c in np.unique(y)]
    return 1 - sum(p ** 2 for p in class_probs)

# Split dataset based on a feature and a threshold
def split_dataset(X, y, feature_index, threshold):
    """
    Split dataset into two subsets: one where the feature value is <= threshold, 
    and one where the feature value is > threshold.
    """
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# Find the best split for a dataset
def best_split(X, y):
    """
    Find the best split based on minimizing Gini index.
    """
    best_gini = float("inf")
    best_split = None
    for feature_index in range(X.shape[1]):  # Iterate over each feature
        thresholds = np.unique(X[:, feature_index])  # Unique values of the feature
        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
            gini_left = gini_index(y_left)
            gini_right = gini_index(y_right)
            gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right  # Weighted average
            if gini < best_gini:
                best_gini = gini
                best_split = (feature_index, threshold)
    return best_split

# Decision Tree Node
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Feature to split on
        self.threshold = threshold  # Threshold for splitting
        self.left = left  # Left child (subset of data where feature <= threshold)
        self.right = right  # Right child (subset of data where feature > threshold)
        self.value = value  # If leaf node, stores the predicted class label

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # Maximum depth of the tree
        self.tree = None  # Root node of the tree

    # Fit the tree on the training data
    def fit(self, X, y):
        self.tree = self._fit(X, y, depth=0)

    # Recursive function to build the tree
    def _fit(self, X, y, depth):
        # If the data is pure or we've reached max depth, return a leaf node
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            return DecisionTreeNode(value=np.argmax(np.bincount(y.flatten())))  # Flatten y to 1D array

        # Find the best split for the data
        feature_index, threshold = best_split(X, y)
        X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
        
        # Recursively build the left and right subtrees
        left_node = self._fit(X_left, y_left, depth + 1)
        right_node = self._fit(X_right, y_right, depth + 1)

        # Return the node
        return DecisionTreeNode(feature_index, threshold, left_node, right_node)

    # Predict the class label for a new data point
    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    # Recursive function to predict class for a single data point
    def _predict(self, x, node):
        if node.value is not None:  # Leaf node
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

# Create synthetic data (simple binary classification problem)
np.random.seed(42)
X = np.random.uniform(0, 10, size=(100, 1))  # Feature
y = (X > 5).astype(int)  # Label: 1 if X > 5, else 0

# Train the decision tree
tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# Predict labels
y_pred = tree.predict(X)

# Calculate accuracy
accuracy = np.mean(y_pred == y)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualize the tree’s decision boundary
X_test = np.linspace(0, 10, 200).reshape(-1, 1)
y_test = tree.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_test, y_test, color="red", label="Decision Boundary")
plt.title("Decision Tree: Actual vs Predicted")
plt.xlabel("X (Input Feature)")
plt.ylabel("y (Predicted Class)")
plt.legend()
plt.show()











# import numpy as np
# import matplotlib.pyplot as plt

# # Gini Index calculation
# def gini_index(y):
#     """
#     Calculate the Gini index for a set of labels.
#     """
#     class_probs = [np.sum(y == c) / len(y) for c in np.unique(y)]
#     return 1 - sum(p ** 2 for p in class_probs)

# # Split dataset based on a feature and a threshold
# def split_dataset(X, y, feature_index, threshold):
#     """
#     Split dataset into two subsets: one where the feature value is <= threshold, 
#     and one where the feature value is > threshold.
#     """
#     left_mask = X[:, feature_index] <= threshold
#     right_mask = ~left_mask
#     return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

# # Find the best split for a dataset
# def best_split(X, y):
#     """
#     Find the best split based on minimizing Gini index.
#     """
#     best_gini = float("inf")
#     best_split = None
#     for feature_index in range(X.shape[1]):  # Iterate over each feature
#         thresholds = np.unique(X[:, feature_index])  # Unique values of the feature
#         for threshold in thresholds:
#             X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
#             gini_left = gini_index(y_left)
#             gini_right = gini_index(y_right)
#             gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right  # Weighted average
#             if gini < best_gini:
#                 best_gini = gini
#                 best_split = (feature_index, threshold)
#     return best_split

# # Decision Tree Node
# class DecisionTreeNode:
#     def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
#         self.feature_index = feature_index  # Feature to split on
#         self.threshold = threshold  # Threshold for splitting
#         self.left = left  # Left child (subset of data where feature <= threshold)
#         self.right = right  # Right child (subset of data where feature > threshold)
#         self.value = value  # If leaf node, stores the predicted class label

# # Decision Tree Classifier
# class DecisionTree:
#     def __init__(self, max_depth=None):
#         self.max_depth = max_depth  # Maximum depth of the tree
#         self.tree = None  # Root node of the tree

#     # Fit the tree on the training data
#     def fit(self, X, y):
#         self.tree = self._fit(X, y, depth=0)

#     # Recursive function to build the tree
#     def _fit(self, X, y, depth):
#         # If the data is pure or we've reached max depth, return a leaf node
#         if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
#             return DecisionTreeNode(value=np.argmax(np.bincount(y)))

#         # Find the best split for the data
#         feature_index, threshold = best_split(X, y)
#         X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
        
#         # Recursively build the left and right subtrees
#         left_node = self._fit(X_left, y_left, depth + 1)
#         right_node = self._fit(X_right, y_right, depth + 1)

#         # Return the node
#         return DecisionTreeNode(feature_index, threshold, left_node, right_node)

#     # Predict the class label for a new data point
#     def predict(self, X):
#         return np.array([self._predict(x, self.tree) for x in X])

#     # Recursive function to predict class for a single data point
#     def _predict(self, x, node):
#         if node.value is not None:  # Leaf node
#             return node.value
#         if x[node.feature_index] <= node.threshold:
#             return self._predict(x, node.left)
#         else:
#             return self._predict(x, node.right)

# # Create synthetic data (simple binary classification problem)
# np.random.seed(42)
# X = np.random.uniform(0, 10, size=(100, 1))  # Feature
# y = (X > 5).astype(int)  # Label: 1 if X > 5, else 0

# # Train the decision tree
# tree = DecisionTree(max_depth=3)
# tree.fit(X, y)

# # Predict labels
# y_pred = tree.predict(X)

# # Calculate accuracy
# accuracy = np.mean(y_pred == y)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

# # Visualize the tree’s decision boundary
# X_test = np.linspace(0, 10, 200).reshape(-1, 1)
# y_test = tree.predict(X_test)

# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, color="blue", label="Actual Data")
# plt.plot(X_test, y_test, color="red", label="Decision Boundary")
# plt.title("Decision Tree: Actual vs Predicted")
# plt.xlabel("X (Input Feature)")
# plt.ylabel("y (Predicted Class)")
# plt.legend()
# plt.show()
