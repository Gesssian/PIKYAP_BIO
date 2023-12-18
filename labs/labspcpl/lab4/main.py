import numpy as np


def PCA(X: np.ndarray, n_components: int) -> np.ndarray:
    mean = np.mean(X, axis=0)
    centered_X = X - mean

    cov_matrix = np.cov(centered_X.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:n_components]]

    transformed_X = np.dot(centered_X, top_eigenvectors)

    return transformed_X


X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
              [13, 14, 15, 16]])

X_new = PCA(X, 3)
print(X_new)
