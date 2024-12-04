# Bootstrapping is a statistical technique that involves sampling from a dataset with replacement to create multiple synthetic datasets. These datasets are then used to estimate the distribution of a statistic (e.g., mean, variance) or to create a predictive model. Bootstrapping is primarily used to assess the variability of a statistic when the underlying distribution is unknown or when the sample size is small.

# In the context of machine learning, bootstrapping is often used as a part of ensemble methods like Bagging and Random Forest. It helps to create multiple diverse training sets from the original data, allowing a more robust model by reducing variance.

# Key Concepts:
# Sampling with Replacement: In bootstrapping, you create new datasets by randomly sampling from the original dataset with replacement. This means some data points may be repeated in a new sample, while others may be omitted.
# Multiple Samples: Multiple datasets are created through bootstrapping. Each sample can be used to train a separate model.
# Estimation of Statistics: After training models on different bootstrapped samples, predictions are made, and the results are aggregated to provide a more accurate estimate.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Function to create bootstrap samples
def bootstrap_sampling(X, y, n_samples):
    # Randomly sample n_samples from the original dataset with replacement
    indices = np.random.choice(range(len(X)), size=n_samples, replace=True)
    return X[indices], y[indices]

# Simple Linear Regression Model (from scratch)
class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Adding a bias term by adding a column of ones to the feature matrix
        X_bias = np.c_[np.ones(X.shape[0]), X]
        # Normal equation for linear regression: (X'X)^(-1) X' y
        self.coefficients = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        # Adding a bias term to the input features
        X_bias = np.c_[np.ones(X.shape[0]), X]
        return X_bias @ self.coefficients

# Generate synthetic data for regression
X, y = make_regression(n_samples=200, n_features=1, noise=0.1, random_state=42)

# Create an array to store the predictions from each bootstrap sample
n_iterations = 100  # Number of bootstrap samples
predictions = np.zeros((n_iterations, X.shape[0]))

# Perform bootstrapping and fit models
for i in range(n_iterations):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = bootstrap_sampling(X, y, len(X))
    
    # Train a Linear Regression model on the bootstrap sample
    model = LinearRegression()
    model.fit(X_bootstrap, y_bootstrap)
    
    # Make predictions on the original data
    predictions[i] = model.predict(X)

# Aggregate predictions (mean and confidence interval)
mean_prediction = predictions.mean(axis=0)
lower_bound = np.percentile(predictions, 2.5, axis=0)  # 2.5th percentile (lower bound of confidence interval)
upper_bound = np.percentile(predictions, 97.5, axis=0)  # 97.5th percentile (upper bound of confidence interval)

# Plot the results
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, mean_prediction, color='red', label='Bootstrap Mean Prediction')
plt.fill_between(X.flatten(), lower_bound, upper_bound, color='gray', alpha=0.2, label='95% Confidence Interval')
plt.title('Bootstrapping for Linear Regression')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()
