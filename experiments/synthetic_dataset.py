import numpy as np
import matplotlib.pyplot as plt

# Create the minority points
from matplotlib import pyplot

from utils.utils_py import generate_synthetic_dataset

n_minority = 250
x_min = np.linspace(0, 1000, n_minority)
y1_min = x_min + 2
y2_min = x_min - 2
X_min = np.hstack([x_min, x_min])
Y_min = np.hstack([y1_min, y2_min])
# Create the majority points
n_majority = 2500
x_max = np.linspace(0, 1000, n_majority)
y1_max = x_max
y2_max = y1_max + (np.random.rand(n_majority) - 0.5) * 2
X_max = np.hstack([x_max, x_max])
Y_max = np.hstack([y1_max, y2_max])
# Plot the points
plt.plot(X_min[:20], Y_min[:20], 'ro', ms=4, label='minority')
plt.plot(X_min[n_minority:n_minority + 20], Y_min[n_minority:n_minority + 20], 'ro', ms=4, label='minority')
plt.plot(X_max[:100], Y_max[:100], 'bo', ms=2, label='majority')
plt.plot(X_max[n_majority:n_majority + 100], Y_max[n_majority:n_majority + 100], 'bo', ms=2, label='majority')
plt.title('Example: Advantage Gamma')
plt.legend()
plt.show()

# Plot the points
plt.plot(X_min, Y_min, 'ro', ms=4, label='minority')
plt.plot(X_max, Y_max, 'bo', ms=2, label='majority')
plt.title('Example: Advantage Gamma')
plt.legend()
plt.show()

x, y = generate_synthetic_dataset()
x_new = x[:40]
y_new = y[:40]
for i in range(2):
    samples_ix = np.where(y_new == i)
    pyplot.scatter(x_new[samples_ix, 0], x_new[samples_ix, 1], s=1)

x = x[500:600]
y = y[500:600]
for i in range(2):
    samples_ix = np.where(y == i)
    pyplot.scatter(x[samples_ix, 0], x[samples_ix, 1], s=1)


pyplot.show()
