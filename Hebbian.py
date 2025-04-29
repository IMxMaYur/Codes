import numpy as np

# Inputs and desired outputs
X = np.array([[1, -1], [-1, 1], [1, 1]])
y = np.array([1, -1, 1])

# Hebbian rule: Δw = η * x * y
weights = np.zeros(2)
learning_rate = 0.1

for xi, yi in zip(X, y):
    weights += learning_rate * xi * yi

print("Learned Weights:", weights)
