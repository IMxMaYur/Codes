import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample high-dimensional data
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7]])

# PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

print("Original shape:", X.shape)
print("Reduced shape:", X_pca.shape)

# Visualize
plt.scatter(X[:, 0], X[:, 1], label='Original Data')
plt.title("Original Data (Before PCA)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
