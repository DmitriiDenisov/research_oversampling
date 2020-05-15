import numpy as np

from experiments.func_for_experiments import generate_random_point_nd
from utils.tests_utils import test_points_on_line
from utils.utils_py import get_vector_two_points, generate_gamma, generate_point_on_line

NUM_RAND_POINTS = 20
N_FEATURES = 10


def main():
    two_points_nd = generate_random_point_nd(num_points=2, n=N_FEATURES)
    assert two_points_nd.shape[0] == 2
    assert two_points_nd.shape[1] == N_FEATURES
    # print(two_points_nd)
    # get vector for these two points
    v = get_vector_two_points(two_points_nd)
    # print(v)

    gamma_coeff = generate_gamma()

    generated_points = generate_point_on_line(two_points_nd[0], v, gamma_coeff)[np.newaxis, :]

    for i in range(NUM_RAND_POINTS - 1):
        gamma_coeff = generate_gamma()
        generated_points = np.concatenate(
            (generate_point_on_line(two_points_nd[0], v, gamma_coeff)[np.newaxis, :], generated_points),
            axis=0)

    assert generated_points.shape[0] == NUM_RAND_POINTS
    assert generated_points.shape[1] == N_FEATURES
    assert test_points_on_line(two_points_nd[0], two_points_nd[1], generated_points)


if __name__ == '__main__':
    main()
