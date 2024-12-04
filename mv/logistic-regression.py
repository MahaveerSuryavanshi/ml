import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_samples = 100
X = np.random.uniform(0, 10, size=(n_samples, 1))  # Features
y = (X > 5).astype(int) + np.random.choice([0, 1], size=(n_samples, 1), p=[0.8, 0.2])  # Labels with noise

# Add bias term to X
X_bias = np.hstack((np.ones((n_samples, 1)), X))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    return -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(np.dot(X, theta))
        gradients = (1 / m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradients
    return theta

# Initialize parameters
theta = np.zeros((X_bias.shape[1], 1))
learning_rate = 0.1
iterations = 100000

# Train the model
theta = gradient_descent(X_bias, y, theta, learning_rate, iterations)

# Predictions
def predict(X, theta):
    return sigmoid(np.dot(X, theta)) >= 0.5

y_pred = predict(X_bias, theta)

# Accuracy
accuracy = np.mean(y_pred == y)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, sigmoid(np.dot(X_bias, theta)), color="red", label="Prediction Curve")
plt.axhline(0.5, color="green", linestyle="--", label="Threshold (0.5)")
plt.title("Logistic Regression: Actual vs Predicted")
plt.xlabel("X (Input Feature)")
plt.ylabel("Probability")
plt.legend()
plt.show()

# Predict new input values
new_X = np.array([[3], [7], [5]])  # Example new data points
new_X_bias = np.hstack((np.ones((new_X.shape[0], 1)), new_X))
new_predictions = predict(new_X_bias, theta)

print("New Input Values and Predictions:")
for i, (input_value, pred) in enumerate(zip(new_X.flatten(), new_predictions.flatten())):
    print(f"Input: {input_value}, Prediction: {'Class 1' if pred else 'Class 0'}")
