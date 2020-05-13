import itertools
from collections import defaultdict

import pandas as pd
import numpy as np

from imblearn.datasets import fetch_datasets

import os

from sklearn.tree import DecisionTreeClassifier

from utils.keras_utils import f1

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

from keras import Sequential
from keras.layers import Dense

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from tqdm import tqdm

from utils.tests_utils import test_points_on_line


# from numpy.random import seed
# seed(1)
# import tensorflow
# tensorflow.random.set_seed(1)

def add_metainfo_dataset(dict_metrics, dataset, num_ones, num_zeros, aug_data, n_neigh, algo):
    dict_metrics['NAME_Dataset'] = '{}_{}'.format(dataset, aug_data)
    dict_metrics['NUM_elements'] = num_ones + num_zeros
    dict_metrics['minority_perc'] = num_ones / (num_ones + num_zeros)
    dict_metrics['Generated_points'] = num_zeros - num_ones
    if isinstance(algo, Sequential):
        dict_metrics['Algo'] = 'NN'
    elif isinstance(algo, RandomForestClassifier):
        dict_metrics['Algo'] = 'RF'
    elif isinstance(algo, SVC):
        dict_metrics['Algo'] = 'SVM'
    elif isinstance(algo, DecisionTreeClassifier):
        dict_metrics['Algo'] = 'DT'

    dict_metrics['N_neigh'] = n_neigh
    return dict_metrics


def add_metadata(df_result, k, theta, success, seed):
    # s2 = pd.Series([Nan, Nan, Nan, Nan], index=['A', 'B', 'C', 'D'])
    # result = df1.append(s2)
    df_result = df_result.append(pd.Series(), ignore_index=True)
    dict_temp = dict()
    dict_temp['NAME_Dataset'] = 'K'
    dict_temp['Algo'] = k
    df_result = df_result.append(dict_temp, ignore_index=True)

    dict_temp = dict()
    dict_temp['NAME_Dataset'] = 'Theta'
    dict_temp['Algo'] = theta
    df_result = df_result.append(dict_temp, ignore_index=True)

    dict_temp = dict()
    dict_temp['NAME_Dataset'] = 'Number of success'
    dict_temp['Algo'] = success
    df_result = df_result.append(dict_temp, ignore_index=True)

    dict_temp = dict()
    dict_temp['NAME_Dataset'] = 'Random Seed'
    dict_temp['Algo'] = seed
    df_result = df_result.append(dict_temp, ignore_index=True)

    return df_result


def distance(x1, x2):
    return np.linalg.norm(x1 - x2)


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
    # if (s > 20):  # заглушка пока что
    #    s = 20
    # s = s / 20
    return s


def get_metrics(y_test, y_pred, aug_data, print_metrics=False):
    f1 = f1_score(y_test.to_numpy().flatten(), y_pred)
    pr = precision_score(y_test.to_numpy().flatten(), y_pred, zero_division=0)
    re = recall_score(y_test.to_numpy().flatten(), y_pred)
    auc_pr = average_precision_score(y_test.to_numpy().flatten(), y_pred)

    dict_ans = {'f1_score': 0 if f1 is None else f1,
                'precision': 0 if pr is None else pr,
                'recall': 0 if re is None else re,
                'AUC_PR': 0 if auc_pr is None else auc_pr}
    """
    if aug_data == 'gamma':
        dict_ans = {'f1_score_gamma': 0 if f1 is None else f1,
                    'precision_gamma': 0 if pr is None else pr,
                    'recall_gamma': 0 if re is None else re,
                    'AUC_PR_gamma': 0 if auc_pr is None else auc_pr}
    elif aug_data == 'no':
        dict_ans = {'f1_score': 0 if f1 is None else f1,
                    'precision': 0 if pr is None else pr,
                    'recall': 0 if re is None else re,
                    'AUC_PR': 0 if auc_pr is None else auc_pr}
    elif aug_data == 'smote':
        dict_ans = {'f1_score_smote': 0 if f1 is None else f1,
                    'precision_smote': 0 if pr is None else pr,
                    'recall_smote': 0 if re is None else re,
                    'AUC_PR_smote': 0 if auc_pr is None else auc_pr}
    """

    if print_metrics:
        print('F1_Score:', f1)
        print('Precision:', pr)
        print('Recall:', re)
        print('AUC_PR:', auc_pr)
    # ans = np.array([f1, pr, re, auc_pr])
    # ans[np.isnan(ans)] = 0
    return dict_ans


# Generates new point on line between two initial points
def generate_point_on_line(start_point, v, gamma_coeff):
    return start_point + v * gamma_coeff


# Generates n random points in hypercube [xy_min; xy_max]
def generate_random_point_nd(num_points=2, n=10, min_=0, max_=10):
    xy_min = [min_] * n
    xy_max = [max_] * n
    data = np.random.uniform(low=xy_min, high=xy_max, size=(num_points, n))
    return data


def generate_points_for_n_minority(minority_points, num_to_add, n_neighbors, k, theta, aug_data='gamma', tol=0.0000001,
                                   testing=False):
    n_features = minority_points.shape[1]
    dict_ans = defaultdict(lambda: np.array([]).reshape(0, n_features))

    if n_neighbors != -1:
        neigh = NearestNeighbors()
        neigh.fit(minority_points)
        all_picked_x = np.random.randint(minority_points.shape[0], size=num_to_add)
        dist, all_neighs = neigh.kneighbors(minority_points[all_picked_x, :],
                                            n_neighbors, return_distance=True)
        assert dist.shape[0] == num_to_add and all_neighs.shape[0] == num_to_add and len(all_picked_x) == num_to_add
        dist, all_neighs = dist[:, 1:], all_neighs[:, 1:]
        # assert np.all(dist > 0)
        #   print('Retrain Neighbours')
        #    dist, all_neighs = neigh.kneighbors(minority_points[all_picked_x, :],
        #                                        n_neighbors+1, return_distance=True)
        #    assert dist.shape[0] == num_to_add and all_neighs.shape[0] == num_to_add and len(all_picked_x) == num_to_add
        #    dist, all_neighs = dist[:, 2:], all_neighs[:, 2:]
        #    assert np.all(dist > 0)

        all_picked_neighs = np.random.choice(range(len(all_neighs[1])), num_to_add, replace=True)
        assert len(all_picked_neighs) == num_to_add

        all_pairs = []
        i = 0
        for picked_x, picked_neigh in zip(all_picked_x, all_picked_neighs):
            all_pairs.append([picked_x, all_neighs[i][picked_neigh]])
            if testing:
                assert abs(
                    distance(minority_points[picked_x], minority_points[all_neighs[i][picked_neigh]]) - dist[i][
                        picked_neigh]) < tol
            i += 1
        all_pairs = np.array(all_pairs)
        assert all_pairs.shape[0] == num_to_add and all_pairs.shape[1] == 2

    else:
        # Choose random pairs with repetition:
        all_pairs = np.array(list(itertools.combinations(range(len(minority_points)), r=2)))
        rand_idx = np.random.choice(range(len(all_pairs)), num_to_add, replace=True)
        assert rand_idx.shape == (num_to_add,)
        # print(rand_idx)
        # print(all_comb)
        all_pairs = all_pairs[rand_idx]
    # assert random_choice_minority.shape[0] == num_to_add and random_choice_minority.shape[1] == 2
    for i, (idx1, idx2) in enumerate(all_pairs):
        v = get_vector_two_points([minority_points[idx1], minority_points[idx2]])
        if aug_data == 'gamma':
            # gamma_coeff = generate_gamma()
            gamma_coeff = generate_gamma_negative(k=k, theta=theta)
            generated_point = generate_point_on_line(minority_points[idx1], v, gamma_coeff)
        elif aug_data == 'smote+normal':
            uniform_coeff = np.random.uniform(0, 1, 1)[0]
            generated_point = generate_point_on_line(minority_points[idx1], v, uniform_coeff)
            generated_point = \
                np.random.multivariate_normal(mean=generated_point, cov=np.eye(generated_point.shape[0], dtype=int),
                                              size=1)[0]
        else:
            generated_point = None
            raise AssertionError("Unexpected value")

        minority_points = np.concatenate((minority_points, generated_point[np.newaxis, :]), axis=0)
        dict_ans[tuple(all_pairs[i])] = np.vstack([dict_ans[tuple(all_pairs[i])], generated_point])

    return minority_points, dict_ans  # return concatenated initial+generated points and dict for testing


def max_pdf_gamma(k, theta):
    return (k - 1) * theta


def generate_gamma_negative(k=1 / 8, theta=2.):
    s = np.random.gamma(k, theta, 1)[0]
    s = s - max_pdf_gamma(k, theta)  # shift by X axis
    # if (s > 20):  # заглушка пока что
    #     s = 20
    # s = s / 20
    return s


def get_dataset_pd(name):
    if name == 'synthetic':
        x, y = generate_synthetic_dataset()
        return pd.DataFrame(x), y
    X = pd.DataFrame(fetch_datasets()[name]['data'])
    target = pd.DataFrame(fetch_datasets()[name]['target']).replace(-1, 0)
    assert target.shape[0] == X.shape[0]
    return X, target


def aug_train(X_temp, n_neighbors, k, theta, aug_data, testing=False):
    # Подавать внутрь датафрейм X_temp с колонкой y
    num_zeros = X_temp[X_temp['y'] == 0].to_numpy().shape[0]
    num_ones = X_temp[X_temp['y'] == 1].to_numpy().shape[0]

    num_add = num_zeros - num_ones
    minority_points = X_temp[X_temp['y'] == 1].drop('y', 1).to_numpy()

    minority_points, dict_ans = generate_points_for_n_minority(minority_points, num_add, n_neighbors, k, theta,
                                                               aug_data=aug_data)
    assert minority_points.shape[0] == X_temp[X_temp['y'] == 1].to_numpy().shape[0] + num_add
    assert num_zeros == minority_points.shape[0]

    # testing:
    if testing:
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


def get_NN(X):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1])
    # model.fit(X, y, epochs=100, verbose=0)
    return model


def get_number_success(df, index=4, step=4, num_modes=7):
    success = 0
    index_init = index
    try:
        index_start = index
        while index < df.shape[0]:
            if df.iloc[index]['f1_score'] > df.iloc[index + step]['f1_score']:
                success += 1
            index += 1
            if index - index_start == index_init:
                index += num_modes * index_init - index_init
                index_start = index
    except IndexError:
        success = np.nan
    return success


def generate_synthetic_dataset():
    # Create the minority points
    n_minority = 250
    x_min = np.linspace(0, 1000, n_minority)
    y1_min = x_min + 2
    y2_min = x_min - 2
    X_min = np.hstack([x_min, x_min])
    Y_min = np.hstack([y1_min, y2_min])
    minority = np.concatenate([X_min[np.newaxis, :], Y_min[np.newaxis, :]], axis=0).T

    # Create the majority points
    n_majority = 2500
    x_max = np.linspace(0, 1000, n_majority)
    y1_max = x_max
    y2_max = y1_max + (np.random.rand(n_majority) - 0.5) * 2
    X_max = np.hstack([x_max, x_max])
    Y_max = np.hstack([y1_max, y2_max])
    majority = np.concatenate([X_max[np.newaxis, :], Y_max[np.newaxis, :]], axis=0).T

    x = np.concatenate([minority, majority], axis=0)
    y = np.array([1] * 2 * n_minority + [0] * 2 * n_majority)
    assert x.shape[0] == len(y)

    return x, y
