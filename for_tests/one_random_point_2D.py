import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from utils.utils import generate_random_point, get_vector_two_points, generate_gamma, generate_point_on_line

# Generate two Random point2:
two_points = generate_random_point(n=2, xy_min=[0, 0], xy_max=[10, 20])
print(two_points)
# get vector for these two points
v = get_vector_two_points(two_points)
print(v)

# JUST CHECKING how Gamma dist is look like
N_points = 100000
n_bins = 20

shape, scale = 1., 3.  # mean=4, std=2*sqrt(2)
s = np.random.gamma(shape, scale, 10000)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs[0].hist(s, bins=n_bins)
axs[1].hist(s, bins=2 * n_bins)
plt.show()

gamma_coeff = generate_gamma()
generated_point = generate_point_on_line(two_points[0], v, gamma_coeff)
# For visualazion:
initial_points_and_generated = np.concatenate((two_points, generated_point[np.newaxis, :]), axis=0)

initial_x, initial_y = map(list, zip(*two_points))
generated_x, generated_y = map(list, zip(*generated_point[np.newaxis, :]))

plt.plot(initial_x, initial_y, 'o', color='blue')
plt.plot(generated_x, generated_y, 'o', color='red')
plt.show()
