import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Linear SVM Classifier (from scratch)
class SVM:
    def __init__(self, learning_rate=0.001, regularization_param=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.reg_param = regularization_param
        self.max_iterations = max_iterations

    def fit(self, X, y):
        # Number of samples and features
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient Descent for Optimization
        for _ in range(self.max_iterations):
            for i in range(n_samples):
                if y[i] * (np.dot(X[i], self.w) + self.b) < 1:
                    # Update rule for misclassified points
                    self.w -= self.learning_rate * (2 * self.reg_param * self.w - np.dot(X[i], y[i]))
                    self.b -= self.learning_rate * y[i]
                else:
                    # Update rule for correctly classified points
                    self.w -= self.learning_rate * 2 * self.reg_param * self.w

    def predict(self, X):
        # Predict the class label for each data point
        return np.sign(np.dot(X, self.w) + self.b)

# Create a simple 2D dataset using sklearn with corrected parameters
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
y = 2 * y - 1  # Convert labels to -1 and 1 for SVM

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the SVM
svm = SVM(learning_rate=0.01, regularization_param=0.1, max_iterations=1000)
svm.fit(X_train, y_train)

# Predictions on the test set
y_pred = svm.predict(X_test)

# Calculate the accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualization
plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label="Training Data")

# Plot the support vectors (misclassified or closest points to hyperplane)
support_vector_mask = (y_train * (np.dot(X_train, svm.w) + svm.b)) < 1
plt.scatter(X_train[support_vector_mask, 0], X_train[support_vector_mask, 1], facecolors='none', edgecolors='black', s=100, label="Support Vectors")

# Plot the decision boundary (hyperplane)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Plot the decision boundary (line)
plt.plot([x_min, x_max], [-(svm.w[0] * x_min + svm.b) / svm.w[1], -(svm.w[0] * x_max + svm.b) / svm.w[1]], color='red', label="Decision Boundary")

# Labels and legend
plt.title("SVM Classifier with Support Vectors and Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
