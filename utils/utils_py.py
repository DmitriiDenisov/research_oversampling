import itertools
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from imblearn.datasets import fetch_datasets
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


def add_metainfo_dataset(dict_metrics: dict, dataset: str, num_ones: int, num_zeros: int, aug_data: str, n_neigh: int,
                         algo: object) -> dict:
    """
    Adds meta info to each row of DataFrame with results such as: name of dataset, num elements and so on
    :param dict_metrics: dict, row of DataFrame as dict
    :param dataset: str, name of Dataset
    :param num_ones: int, number of minority points in dataset
    :param num_zeros: int, number of majority points in dataset
    :param aug_data: str, mode of augmentation of data
    :param n_neigh: int, number of neighbours for augmentation
    :param algo: object, ML model
    :return: dict, dictionary which is new row for the resulting DataFrame
    """
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


def get_clf(classname: str, num_f: int):
    """
    Function defines and returns algorithm
    :param classname: str, name of algorithm
    :param num_f: int, number of features in dataset
    :return: object, Algorithm
    """
    if classname == 'DT':
        return DecisionTreeClassifier()
    elif classname == 'RF':
        return RandomForestClassifier(n_estimators=50)
    elif classname == 'SVC':
        return SVC(gamma='auto')
    elif classname == 'NN':
        return get_NN(num_f)


def add_metadata(df_result: pd.DataFrame, k: float, theta: float, success: int, seed: int) -> pd.DataFrame:
    """
    Adds meta info to resulting DataFrame namely K, Theta, Number of Successes and Random seed
    :param df_result: pd.DataFrame, Resulting DataFrame with all metrics
    :param k: float, k param for gamma oversampling
    :param theta: float, theta param for gamma oversampling
    :param success: int, number of successes when gamma oversampling outperformed another method
    :param seed: int, random seed
    :return: pd.DataFrame, Resulting DataFrame with added metadata
    """
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


def get_vector_two_points(two_points: np.array) -> np.array:
    """
    Returns vector of direction
    :param two_points: np.array, two points
    :return: np.array, vector of direction from point two_points[0] to two_points[1]
    """
    return two_points[1] - two_points[0]


def max_pdf_gamma(k: float, theta: float) -> float:
    """
    Returns argmax value of Gamma dist
    :param k: float, k param from Gamma dist
    :param theta: float, theta param from Gamma dist
    :return: float, argmax of Gamma dist
    """
    return (k - 1) * theta


def generate_gamma_negative(k: float = 1 / 8, theta: float = 2.) -> float:
    """
    Generates gamma dist with shift on argmax value, i.e peak of our gamma dist is point_1
    :param k: float, k param from Gamma dist
    :param theta: float, theta param from Gamma dist
    :return: float, generated coefficient with shifted gamma dist
    """
    s = np.random.gamma(k, theta, 1)[0]
    s = s - max_pdf_gamma(k, theta)
    # if (s > 20):  # normalization
    #     s = 20
    # s = s / 20
    return s


def generate_gamma() -> float:
    """
    Generates Gamma distributed value
    :return: float,  generated coefficient with gamma dist
    """
    shape, scale = 1., 3.
    s = np.random.gamma(shape, scale, 1)[0]
    # if (s > 20):  # заглушка пока что
    #    s = 20
    # s = s / 20
    return s


def get_metrics(y_test: pd.Series, y_pred: np.array, print_metrics: bool = False) -> dict:
    """
    Calculates metrics f1, precision, recall, AUC
    :param y_test: pd.Series, y_true values
    :param y_pred: np.array, y_pred values
    :param print_metrics: bool, if print results or no
    :return: dict, dictionary with calculated metrics
    """
    f1 = f1_score(y_test.to_numpy().flatten(), y_pred)
    pr = precision_score(y_test.to_numpy().flatten(), y_pred, zero_division=0)
    re = recall_score(y_test.to_numpy().flatten(), y_pred)
    auc_pr = average_precision_score(y_test.to_numpy().flatten(), y_pred)

    dict_ans = {'f1_score': 0 if f1 is None else f1,
                'precision': 0 if pr is None else pr,
                'recall': 0 if re is None else re,
                'AUC_PR': 0 if auc_pr is None else auc_pr}

    if print_metrics:
        print('F1_Score:', f1)
        print('Precision:', pr)
        print('Recall:', re)
        print('AUC_PR:', auc_pr)
    return dict_ans


def generate_point_on_line(start_point: np.array, v: np.array, gamma_coeff: float) -> np.array:
    """
    Generates new point on line between two initial points
    :param start_point: np.array, initial point
    :param v: np.array, vector direction from point_1 to point_2
    :param gamma_coeff: float, coefficient generated from gamma dist
    :return: np.array, new generated point on line between point_1 and point_2
    """
    return start_point + v * gamma_coeff


def generate_points_for_n_minority(minority_points: np.array, num_to_add: int, n_neighbors: int, k: float, theta: float,
                                   aug_data: str = 'gamma') -> (
        np.array, dict):
    """
    Makes augmentation of minority points
    :param minority_points: np.array, array of all minority points
    :param num_to_add: int, number of points needed to be generated
    :param n_neighbors: int, number of neighbours
    :param k: float, k param from Gamma dist
    :param theta: float, theta param from Gamma dist
    :param aug_data: str, mode of oversampling/undersampling
    :return: (np.array, dict), returns concatenated initial + generated points and dict for testing
    """
    n_features = minority_points.shape[1]
    dict_ans = defaultdict(lambda: np.array([]).reshape(0, n_features))

    if n_neighbors != -1:
        neigh = NearestNeighbors()
        neigh.fit(minority_points)
        all_picked_x = np.random.randint(minority_points.shape[0], size=num_to_add)
        dist, all_neighs = neigh.kneighbors(minority_points[all_picked_x, :],
                                            n_neighbors, return_distance=True)
        assert all_neighs.shape[1] == n_neighbors
        assert dist.shape[0] == num_to_add and all_neighs.shape[0] == num_to_add and len(all_picked_x) == num_to_add
        dist, all_neighs = dist[:, 1:], all_neighs[:, 1:]  # remove first because closest Neighbour is the point itself
        assert np.all(dist > 0)

        all_picked_neighs = np.random.choice(range(n_neighbors - 1), num_to_add, replace=True)
        assert len(all_picked_neighs) == num_to_add

        # Get pairs: [[picked_point, neigh], [...], ...]
        # https://stackoverflow.com/questions/23435782/numpy-selecting-specific-column-index-per-row-by-using-a-list-of-indexes
        all_pairs = np.array(list(zip(all_picked_x, all_neighs[np.arange(len(all_neighs)), all_picked_neighs])))
        assert all_pairs.shape[0] == num_to_add and all_pairs.shape[1] == 2
    else:
        # Choose random pairs with repetition:
        all_pairs = np.array(list(itertools.combinations(range(len(minority_points)), r=2)))
        rand_idx = np.random.choice(range(len(all_pairs)), num_to_add, replace=True)
        assert rand_idx.shape == (num_to_add,)
        all_pairs = all_pairs[rand_idx]
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

    return minority_points, dict_ans


def get_dataset_pd(name: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Function Parses dataset from imblearn fetch_dataset() function or generates synthetic dataset
    :param name: str, name of Dataset
    :return: (pd.DataFrame, pd.DataFrame) Returns dataset X and target variable y
    """
    if name == 'synthetic':
        x, y = generate_synthetic_dataset()
        return pd.DataFrame(x), y
    X = pd.DataFrame(fetch_datasets()[name]['data'])
    target = pd.DataFrame(fetch_datasets()[name]['target']).replace(-1, 0)
    assert target.shape[0] == X.shape[0]
    return X, target


def aug_train(X_temp: pd.DataFrame, n_neighbors: int, k: float, theta: float, aug_data: str,
              testing: bool = False) -> pd.DataFrame:
    """
    Augmentation of train dataset
    :param X_temp: pd.DataFrame,
    :param n_neighbors: int, number of neighbours
    :param k: float, k param from Gamma dist
    :param theta: float, k param from Gamma dist
    :param aug_data: str, mode of oversampling/undersampling
    :param testing: bool, if needed to make tests or not
    :return: pd.DataFrame, augmented training dataset
    """
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


def get_NN(num_f: int) -> Sequential:
    """
    Creates instance of Neural Network
    :param num_f: int, number of features in dataset
    :return: Sequential, Neural Netwrok
    """
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=num_f))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1])
    # model.fit(X, y, epochs=100, verbose=0)
    return model


def get_number_success(df: pd.DataFrame, index: int = 4, step: int = 4, num_modes: int = 7) -> int:
    """
    Calculates number of successes, i.e when gamma augmentation outperformed another method
    :param df: pd.DataFrame, DataFrame with all results and metrics
    :param index: int, index where Gamma metrics start
    :param step: int, distance from Gamma metrics to another method
    :param num_modes: int, number of augmentation methods
    :return: int, number of successes
    """
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


def generate_synthetic_dataset() -> (np.array, np.array):
    """
    Generates synthetic dataset
    :return: (np.array, np.array)
    """
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
