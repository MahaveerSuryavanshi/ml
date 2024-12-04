import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the SVM class (same as previous implementation)
class SVM:
    def __init__(self, learning_rate=0.01, regularization_param=0.1, max_iters=1000):
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.max_iters = max_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.max_iters):
            for i in range(X.shape[0]):
                if y[i] * (np.dot(X[i], self.w) + self.b) < 1:
                    self.w -= self.learning_rate * (2 * self.regularization_param * self.w - np.dot(X[i], y[i]))
                    self.b -= self.learning_rate * y[i]
                else:
                    self.w -= self.learning_rate * 2 * self.regularization_param * self.w

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Function to load data from a CSV file
def load_data(file_path, feature_columns, target_column):
    """
    Loads data from a CSV file and prepares it for the SVM model.
    :param file_path: Path to the CSV file
    :param feature_columns: List of column names or indices for the features
    :param target_column: Name or index of the target column
    :return: Features (X), Target (y)
    """
    # Load the dataset from CSV
    data = pd.read_csv(file_path)

    # Extract features and target based on the provided column names
    X = data[feature_columns].values  # Features (inputs)
    y = data[target_column].values  # Target (output)

    # Convert target labels to -1 and 1 for binary classification (if needed)
    y = np.where(y == 0, -1, y)  # Change 0 class to -1 if needed

    return X, y

# Visualization function for the decision boundary (same as previous implementation)
def plot_decision_boundary(X, y, model):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.title("SVM: Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Main function to run the SVM
def main():
    # File path to your CSV data
    file_path = 'svm.csv'  # Change this to your CSV file path

    # Specify the feature columns and target column from the CSV
    feature_columns = ['feature1', 'feature2']  # Replace with your actual feature column names or indices
    target_column = 'target'  # Replace with your actual target column name

    # Load data from the CSV
    X, y = load_data(file_path, feature_columns, target_column)

    # Create and train the SVM model
    svm = SVM(learning_rate=0.01, regularization_param=0.1, max_iters=1000)
    svm.fit(X, y)

    # Make predictions on the training data
    y_pred = svm.predict(X)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Plot decision boundary (if it's 2D data)
    if X.shape[1] == 2:  # Only for 2D features
        plot_decision_boundary(X, y, svm)

if __name__ == "__main__":
    main()
