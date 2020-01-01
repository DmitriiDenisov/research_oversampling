import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
from utils.utils import generate_random_point, get_vector_two_points, generate_gamma_negative, generate_point_on_line
import numpy as np

NUM_POINTS = 50

# Generate two Random point2:
two_points = generate_random_point(n=2, xy_min=[0, 0], xy_max=[10, 20])
print(two_points)
# get vector for these two points
v = get_vector_two_points(two_points)
print(v)

gamma_coeff = generate_gamma_negative()

generated_points = generate_point_on_line(two_points[0], v, gamma_coeff)[np.newaxis, :]

for i in range(NUM_POINTS - 1):
    gamma_coeff = generate_gamma_negative()
    generated_points = np.concatenate(
        (generate_point_on_line(two_points[0], v, gamma_coeff)[np.newaxis, :], generated_points),
        axis=0)

initial_x, initial_y = map(list, zip(*two_points))
generated_x, generated_y = map(list, zip(*generated_points))

plt.plot(initial_x, initial_y, 'o', color='blue')
plt.plot(generated_x, generated_y, '.', color='red')
plt.plot(initial_x, initial_y, '-')
plt.show()
