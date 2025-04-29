import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Sample data
X = np.array([[2, 3], [1, 1], [2, 1], [3, 2], [6, 5], [7, 8]])
y = [0, 0, 0, 0, 1, 1]

# Train SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=y)
ax = plt.gca()
xlim = ax.get_xlim()
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(xlim[0], xlim[1])
yy = a * xx - (clf.intercept_[0]) / w[1]
plt.plot(xx, yy, 'k-')
plt.title("SVM Decision Boundary")
plt.show()
