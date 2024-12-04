# Key Concepts:
# Distance Metric: KNN uses distance metrics like Euclidean distance to measure the "closeness" between data points.
# Majority Voting: For classification, the predicted label for a data point is the most common label among the k nearest neighbors.
# Choice of k: The value of k (the number of nearest neighbors) is an important hyperparameter that affects model performance.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from collections import Counter

# K-Nearest Neighbors Classifier (from scratch)
class KNN:
    def __init__(self, k=3):
        self.k = k  # number of neighbors to consider

    def fit(self, X_train, y_train):
        self.X_train = X_train  # Store the training data
        self.y_train = y_train  # Store the labels for training data

    def euclidean_distance(self, point1, point2):
        # Compute the Euclidean distance between two points
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test]  # Predict each test sample
        return np.array(y_pred)

    def _predict(self, x):
        # Calculate distances between the test point and all training points
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort the distances and get indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Majority voting - most common label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Create a simple 2D dataset using sklearn
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the KNN classifier with k=3
knn = KNN(k=3)

# Train the model
knn.fit(X_train, y_train)

# Predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualization
plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label="Training Data")

# Plot the decision boundary (using a mesh grid)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Labels and legend
plt.title("KNN Classifier with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
