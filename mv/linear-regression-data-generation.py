import numpy as np
import matplotlib.pyplot as plt

def generate_linear_data_uniform(n_samples=100, x_range=(0, 10), slope=1.0, intercept=0.0, noise=1.0):
    """
    Generate linear data using uniform distribution.
    """
    X = np.random.uniform(x_range[0], x_range[1], size=(n_samples, 1))
    y = slope * X + intercept + np.random.uniform(-noise, noise, size=(n_samples, 1))
    return X, y

def generate_linear_data_normal(n_samples=100, x_range=(0, 10), slope=1.0, intercept=0.0, noise=1.0):
    """
    Generate linear data using normal distribution.
    """
    X = np.random.uniform(x_range[0], x_range[1], size=(n_samples, 1))
    y = slope * X + intercept + np.random.normal(0, noise, size=(n_samples, 1))
    return X, y

def generate_polynomial_data(n_samples=100, x_range=(0, 10), coefficients=(1.0, -0.5, 0.2), noise=1.0):
    """
    Generate polynomial data with specified coefficients.
    Coefficients correspond to a polynomial like: c2*x^2 + c1*x + c0
    """
    X = np.random.uniform(x_range[0], x_range[1], size=(n_samples, 1))
    y = sum(c * (X ** i) for i, c in enumerate(coefficients)) + np.random.normal(0, noise, size=(n_samples, 1))
    return X, y

# Visualization function
def plot_generated_data(X, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color="blue", label="Generated Data")
    plt.title(title)
    plt.xlabel("X (Input)")
    plt.ylabel("y (Output)")
    plt.legend()
    plt.show()

# Generate and visualize data
n_samples = 100
x_range = (0, 10)

# Linear data (uniform)
X_uniform, y_uniform = generate_linear_data_uniform(n_samples, x_range, slope=2.5, intercept=1.0, noise=1.0)
plot_generated_data(X_uniform, y_uniform, "Linear Data (Uniform Distribution)")

# Linear data (normal)
X_normal, y_normal = generate_linear_data_normal(n_samples, x_range, slope=2.5, intercept=1.0, noise=1.0)
plot_generated_data(X_normal, y_normal, "Linear Data (Normal Distribution)")

# Polynomial data
X_poly, y_poly = generate_polynomial_data(n_samples, x_range, coefficients=(1.0, -0.5, 0.2), noise=1.0)
plot_generated_data(X_poly, y_poly, "Polynomial Data")
