# pytest tests.py -vv
# pytest tests.py -v
import os

from utils.handle_dataset import handle_dataset
from utils.utils import get_dataset_pd


def test_1(capsys, caplog):
    os.system('python3 for_tests/multiple_random_point_ND_generalized.py')


def test_2(capsys, caplog):
    os.system('python3 for_tests/multiple_random_point_ND.py')


def test_3(capsys, caplog):
    os.system('python3 for_tests/one_random_point_ND.py')


def test_4(capsys, caplog):
    dataset = 'abalone'
    INITIAL_FOLDS = 5
    N_NEIGH = 5
    X_temp, y = get_dataset_pd(dataset)
    X_temp['y'] = y
    # Drop duplicates:
    X_temp = X_temp.drop_duplicates()
    y = X_temp[['y']]
    dict_metrics, num_folds = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='no', num_folds=INITIAL_FOLDS)
    dict_metrics, num_folds_gamma = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='gamma',
                                                   num_folds=INITIAL_FOLDS,
                                                   n_neighbours=N_NEIGH)
    dict_metrics, num_folds_smote = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='smote',
                                                   num_folds=INITIAL_FOLDS,
                                                   n_neighbours=N_NEIGH)
