import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
from utils.tests_utils import test_points_on_line_old, test_points_on_line
from utils.utils import get_vector_two_points, generate_gamma, generate_point_on_line, \
    generate_random_point_nd


def main():
    two_points_nd = generate_random_point_nd(num_points=2, n=10)
    assert two_points_nd.shape[0] == 2
    assert two_points_nd.shape[1] == 10
    # print(two_points_nd)
    # get vector for these two points
    v = get_vector_two_points(two_points_nd)
    # print(v)

    gamma_coeff = generate_gamma()
    generated_point = generate_point_on_line(two_points_nd[0], v, gamma_coeff)

    # Testing
    assert test_points_on_line(two_points_nd[0], two_points_nd[1], generated_point)
    assert test_points_on_line_old(two_points_nd[0], two_points_nd[1], generated_point)


if __name__ == '__main__':
    main()
