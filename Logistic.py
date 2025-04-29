import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Plotting
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict_proba(X)[:, 1], color='red')
plt.title("Logistic Regression")
plt.show()
