# pytest tests.py -vv
# pytest tests.py -v
import os
import warnings

from sklearn.ensemble import RandomForestClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from for_tests import multiple_random_point_ND, multiple_random_point_ND_generalized, one_random_point_ND
from utils.handle_dataset import handle_dataset
from utils.utils import get_dataset_pd


def test_1(capsys, caplog):
    # os.system('python3 for_tests/multiple_random_point_ND_generalized.py')
    multiple_random_point_ND_generalized.main()


def test_2(capsys, caplog):
    # os.system('python3 for_tests/multiple_random_point_ND.py')
    multiple_random_point_ND.main()


def test_3(capsys, caplog):
    # os.system('python3 for_tests/one_random_point_ND.py')
    one_random_point_ND.main()


def test_4(capsys, caplog):
    dataset = 'abalone'
    INITIAL_FOLDS = 5
    N_NEIGH = 5
    X_temp, y = get_dataset_pd(dataset)
    X_temp['y'] = y
    # Drop duplicates:
    X_temp = X_temp.drop_duplicates()
    y = X_temp[['y']]

    clf = RandomForestClassifier(n_estimators=50)
    k = 1
    theta = 2

    dict_metrics, num_folds = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='no', num_folds=INITIAL_FOLDS,
                                             n_neighbours=N_NEIGH, clf=clf, k=k, theta=theta)
    dict_metrics, num_folds_gamma = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='gamma',
                                                   num_folds=INITIAL_FOLDS,
                                                   n_neighbours=N_NEIGH, clf=clf, k=k,
                                                   theta=theta)
    dict_metrics, num_folds_smote = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='smote',
                                                   num_folds=INITIAL_FOLDS,
                                                   n_neighbours=N_NEIGH, clf=clf, k=k,
                                                   theta=theta)
