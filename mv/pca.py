# Principal Component Analysis (PCA) from Scratch
# Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms the features of a dataset into a new set of orthogonal features (principal components) that maximize the variance. The first few principal components capture most of the information in the data, which allows us to reduce the dimensionality of the dataset while preserving its variance.

# Key Concepts of PCA:
# Eigenvalues and Eigenvectors: PCA identifies the directions (principal components) in which the data varies the most. These directions are represented by the eigenvectors of the covariance matrix, and the magnitude of the variance along these directions is represented by the eigenvalues.
# Covariance Matrix: The covariance matrix captures the pairwise relationships between the features in the data. PCA computes the eigenvectors of this matrix to find the principal components.
# Dimensionality Reduction: By selecting the top k eigenvectors corresponding to the largest eigenvalues, we reduce the dimensionality of the data while retaining most of the information.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SKPCA

# PCA Algorithm (from scratch)
class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components  # Number of principal components to keep

    def fit(self, X):
        # Standardize the data (important for PCA)
        X = StandardScaler().fit_transform(X)

        # Compute the covariance matrix of the data
        cov_matrix = np.cov(X.T)

        # Compute eigenvalues and eigenvectors of the covariance matrix
        eigvals, eigvecs = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]

        # Select the top 'n_components' eigenvectors
        self.components = eigvecs[:, :self.n_components]

    def transform(self, X):
        # Project the data onto the selected principal components
        X = StandardScaler().fit_transform(X)
        return X.dot(self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Create a synthetic 2D dataset using sklearn (with more than 2 features)
X, _ = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=0, random_state=42)

# Initialize PCA with 2 components (for 2D visualization)
pca = PCA(n_components=2)

# Apply PCA to reduce dimensionality
X_pca = pca.fit_transform(X)

# Visualize the data after PCA transformation
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', edgecolor='k', marker='o', label="PCA transformed data")
plt.title("PCA - 2D Projection of Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# For comparison, use scikit-learn's PCA to perform the same transformation
sklearn_pca = SKPCA(n_components=2)
X_sklearn_pca = sklearn_pca.fit_transform(X)

# Visualize the sklearn PCA results
plt.figure(figsize=(8, 6))
plt.scatter(X_sklearn_pca[:, 0], X_sklearn_pca[:, 1], c='red', edgecolor='k', marker='o', label="sklearn PCA transformed data")
plt.title("sklearn PCA - 2D Projection of Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()
