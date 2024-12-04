import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)  # Random data for input
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relationship with noise

# Initialize parameters
theta_0 = 0  # Intercept
theta_1 = 0  # Slope
learning_rate = 0.01
iterations = 1000
m = len(X)  # Number of samples

# Hypothesis function
def predict(X, theta_0, theta_1):
    return theta_0 + theta_1 * X

# Cost function
def compute_cost(X, y, theta_0, theta_1):
    predictions = predict(X, theta_0, theta_1)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Gradient descent function
def gradient_descent(X, y, theta_0, theta_1, learning_rate, iterations):
    for _ in range(iterations):
        predictions = predict(X, theta_0, theta_1)
        # Compute gradients
        d_theta_0 = (1 / m) * np.sum(predictions - y)
        d_theta_1 = (1 / m) * np.sum((predictions - y) * X)
        # Update parameters
        theta_0 -= learning_rate * d_theta_0
        theta_1 -= learning_rate * d_theta_1
    return theta_0, theta_1

# Train the model
theta_0, theta_1 = gradient_descent(X, y, theta_0, theta_1, learning_rate, iterations)

# Predictions
y_pred = predict(X, theta_0, theta_1)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Prediction")
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("X (Input Feature)")
plt.ylabel("y (Target Value)")
plt.legend()
plt.show()

# Print the final parameters
print(f"Final parameters: theta_0 = {theta_0:.2f}, theta_1 = {theta_1:.2f}")
