import itertools
from collections import defaultdict

import pandas as pd
import numpy as np

from imblearn.datasets import fetch_datasets

from utils.tests_utils import test_points_on_line


# Generates random point in rectangle [xy_min; xy_max]
def generate_random_point(n=2, xy_min=[0, 0], xy_max=[10, 20]):
    data = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
    return data


# Returns directional vector
def get_vector_two_points(two_points):
    return two_points[1] - two_points[0]


# Generates normilized Gamma distributed value
def generate_gamma():
    shape, scale = 1., 3.
    s = np.random.gamma(shape, scale, 1)[0]
    if (s > 20):  # заглушка пока что
        s = 20
    s = s / 20
    return s


# Generates new point on line between two initial points
def generate_point_on_line(start_point, v, gamma_coeff):
    return start_point + v * gamma_coeff


# Generates n random points in hypercube [xy_min; xy_max]
def generate_random_point_nd(num_points=2, n=10, min_=0, max_=10):
    xy_min = [min_] * n
    xy_max = [max_] * n
    data = np.random.uniform(low=xy_min, high=xy_max, size=(num_points, n))
    return data


def generate_points_for_n_minority(minority_points, num_to_add):
    n_features = minority_points.shape[1]
    dict_ans = defaultdict(lambda: np.array([]).reshape(0, n_features))
    # Choose random pairs with repetition:
    all_comb = np.array(list(itertools.combinations(range(len(minority_points)), r=2)))
    rand_idx = np.random.choice(range(len(all_comb)), num_to_add, replace=True)
    assert rand_idx.shape == (num_to_add,)
    # print(rand_idx)
    # print(all_comb)
    all_comb = all_comb[rand_idx]
    # assert random_choice_minority.shape[0] == num_to_add and random_choice_minority.shape[1] == 2
    for i, (idx1, idx2) in enumerate(all_comb):
        v = get_vector_two_points([minority_points[idx1], minority_points[idx2]])
        gamma_coeff = generate_gamma()
        generated_point = generate_point_on_line(minority_points[idx1], v, gamma_coeff)
        minority_points = np.concatenate((minority_points, generated_point[np.newaxis, :]), axis=0)
        dict_ans[tuple(all_comb[i])] = np.vstack([dict_ans[tuple(all_comb[i])], generated_point])

    return minority_points, dict_ans  # return concatenated initial+generated points and dict for testing


def max_pdf_gamma(k, theta):
    return (k - 1) * theta


def generate_gamma_negative():
    k, theta = 3, 2.1
    s = np.random.gamma(k, theta, 1)[0]
    s = s - max_pdf_gamma(k, theta)  # shift by X axis
    if (s > 20):  # заглушка пока что
        s = 20
    s = s / 20
    return s


def get_dataset_pd(name):
    X = pd.DataFrame(fetch_datasets()[name]['data'])
    target = pd.DataFrame(fetch_datasets()[name]['target']).replace(-1, 0)
    assert target.shape[0] == X.shape[0]
    return X, target


def aug_train(X_temp):
    # Подавать внутрь датафрейм X_temp с колонкой y
    num_zeros = X_temp[X_temp['y'] == 0].to_numpy().shape[0]
    num_ones = X_temp[X_temp['y'] == 1].to_numpy().shape[0]

    num_add = num_zeros - num_ones
    minority_points = X_temp[X_temp['y'] == 1].drop('y', 1).to_numpy()

    minority_points, dict_ans = generate_points_for_n_minority(minority_points, num_add)
    assert minority_points.shape[0] == X_temp[X_temp['y'] == 1].to_numpy().shape[0] + num_add
    assert num_zeros == minority_points.shape[0]

    # testing:
    initial_rows = X_temp[X_temp['y'] == 1].to_numpy().shape[0]
    n_points = 0
    for key, points in dict_ans.items():
        for point in points:
            n_points += 1
            assert key[1] <= initial_rows and key[0] <= initial_rows
            assert test_points_on_line(minority_points[key[1]], minority_points[key[0]], point)
    assert n_points == num_add
    assert np.all(np.equal(minority_points[:initial_rows], X_temp[X_temp['y'] == 1].drop('y', 1).to_numpy()))

    X_aug = np.concatenate((X_temp[X_temp['y'] == 0].drop('y', 1).to_numpy(), minority_points), axis=0)
    y_aug = np.array([0] * num_zeros + [1] * num_zeros)
    assert X_aug.shape[0] == 2 * num_zeros
    assert y_aug.shape[0] == X_aug.shape[0]

    df_new = pd.DataFrame(X_aug)
    df_new['y'] = y_aug

    return df_new