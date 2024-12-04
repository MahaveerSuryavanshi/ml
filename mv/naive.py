import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import norm

# Naive Bayes Classifier (from scratch)
class NaiveBayes:
    def __init__(self):
        self.mean = {}
        self.var = {}
        self.class_prob = {}
        self.classes = []

    def fit(self, X_train, y_train):
        # Get unique classes
        self.classes = np.unique(y_train)
        
        # Calculate mean, variance and class probability for each class
        for c in self.classes:
            # Get data points belonging to the class
            X_c = X_train[y_train == c]
            
            # Calculate mean and variance for each feature in this class
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            self.class_prob[c] = len(X_c) / len(X_train)

    def predict(self, X_test):
        # Predict the class label for each sample
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):
        # Calculate the probabilities for each class
        class_probs = {}
        
        for c in self.classes:
            prior_prob = np.log(self.class_prob[c])
            likelihood = np.sum(np.log(self._pdf(c, x)))
            class_probs[c] = prior_prob + likelihood
        
        # Return the class with the highest probability
        return max(class_probs, key=class_probs.get)

    def _pdf(self, c, x):
        # Calculate the probability density function for each feature (assumed Gaussian)
        mean = self.mean[c]
        var = self.var[c]
        return norm.pdf(x, mean, np.sqrt(var))  # Gaussian PDF formula

# Create a simple 2D dataset using sklearn
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Naive Bayes classifier
nb = NaiveBayes()

# Train the model
nb.fit(X_train, y_train)

# Predictions on the test set
y_pred = nb.predict(X_test)

# Calculate the accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# New input prediction
new_data = np.array([[0.5, -0.3]])  # Example input data
new_pred = nb.predict(new_data)
print(f"Predicted class for new input data: {new_pred[0]}")

# Visualization
plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label="Training Data")

# Plot the decision boundary (using a mesh grid)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

# Labels and legend
plt.title("Naive Bayes Classifier with Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
