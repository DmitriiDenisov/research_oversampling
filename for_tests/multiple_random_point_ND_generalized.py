from utils.tests_utils import test_points_on_line
from utils.utils import generate_random_point_nd, generate_points_for_n_minority

NUM_RAND_POINTS = 20
NUM_MINORITY_POINTS = 5
N_FEATURES = 10
NUM_TO_ADD = 15


def main():
    minority_points_nd = generate_random_point_nd(num_points=NUM_MINORITY_POINTS, n=N_FEATURES)
    assert minority_points_nd.shape[0] == NUM_MINORITY_POINTS
    assert minority_points_nd.shape[1] == N_FEATURES
    # print(two_points_nd)
    # get vector for these two points
    # v = get_vector_two_points(N_points_nd)
    # print(v)

    minority_points, dict_ans = generate_points_for_n_minority(minority_points_nd, num_to_add=NUM_TO_ADD, n_neighbors=3,
                                                               k=1, theta=2)

    # testing:
    n_points = 0
    for key, points in dict_ans.items():
        for point in points:
            n_points += 1
            assert test_points_on_line(minority_points_nd[key[1]], minority_points_nd[key[0]], point)
    assert n_points == NUM_TO_ADD


if __name__ == '__main__':
    main()
