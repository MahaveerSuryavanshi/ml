# Boosting Algorithm from Scratch
# Boosting is an ensemble technique where multiple weak models (often decision trees) are trained sequentially, and each new model attempts to correct the errors made by the previous models. Unlike bagging, where models are trained independently, boosting trains models in a sequence, making it more powerful for improving the accuracy of weak learners.

# The most popular boosting algorithm is AdaBoost (Adaptive Boosting), which adjusts the weights of incorrectly classified data points so that subsequent models focus on them.

# Key Concepts of Boosting:
# Weak Learner: A weak learner is a model that performs slightly better than random guessing. Decision trees with limited depth (e.g., decision stumps) are often used as weak learners in boosting.
# Error Calculation: In boosting, after each model is trained, the error is calculated, and the weights of misclassified points are increased to make the next model focus more on those examples.
# Weighted Prediction: Predictions from each weak model are combined with weights, where models with lower error get higher weights in the final prediction.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# AdaBoost Classifier (from scratch)
class AdaBoostClassifier:
    def __init__(self, base_model, n_estimators=50):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize the weights of all samples as equal
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # Train a base model on the weighted dataset
            model = self.base_model()
            model.fit(X, y, sample_weight=sample_weights)

            # Predict the labels for the current model
            y_pred = model.predict(X)

            # Calculate the error rate (weighted error)
            incorrect = (y_pred != y)
            error = np.sum(sample_weights * incorrect) / np.sum(sample_weights)

            # Compute alpha (the model weight based on its error)
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update the weights of misclassified samples
            sample_weights *= np.exp(-alpha * y * y_pred)

            # Normalize sample weights to sum to 1
            sample_weights /= np.sum(sample_weights)

            # Store the model and alpha
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # Aggregate the predictions of all models, weighted by alpha
        final_prediction = np.zeros(X.shape[0])

        for model, alpha in zip(self.models, self.alphas):
            y_pred = model.predict(X)
            final_prediction += alpha * y_pred

        # Predict the class with the majority vote
        return np.sign(final_prediction)

# Create synthetic dataset for classification
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=0, random_state=42)

# Convert labels to -1 and 1 for AdaBoost (AdaBoost assumes labels are -1 or 1)
y = 2 * y - 1

# Initialize AdaBoost classifier with Decision Trees as the base model
adaboost = AdaBoostClassifier(base_model=DecisionTreeClassifier, n_estimators=50)

# Train the AdaBoost model
adaboost.fit(X, y)

# Predict using the trained model
y_pred = adaboost.predict(X)

# Evaluate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy of AdaBoost model: {accuracy * 100:.2f}%")

# Visualize a subset of predictions (if 2D for simplicity)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o', edgecolor='k')
plt.title("AdaBoost Model Predictions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
