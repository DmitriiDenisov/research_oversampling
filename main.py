import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore")

from itertools import product
from os import path
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from utils.handle_dataset import handle_dataset
from utils.constants import *

from utils.utils import get_dataset_pd, add_metainfo_dataset, get_NN, add_metadata, get_number_success

# datasets = [
#             'ozone_level',
#             'mammography',
#    ]

# datasets = ['abalone', 'sick_euthyroid']

list_k_theta = [[0.125, 2.], [1.5, 6.5], [1.7, 7], [1.7, 2], [1, 2], [1, 4]]
for (k, theta) in tqdm(list_k_theta):
    print(k, theta)
    if path.exists('output.xlsx'):
        df_result = pd.read_excel('output.xlsx', index_col=None)
    else:
        df_result = pd.DataFrame(
            columns=COLUMNS)

    for dataset in DATASETS:
        # continue # !!!!!
        print(dataset)
        X_temp, y = get_dataset_pd(dataset)
        classifiers = [get_NN(X_temp), RandomForestClassifier(n_estimators=50), SVC(gamma='auto')]
        # classifiers = [SVC(gamma='auto')]
        assert np.all(np.unique(y) == np.array([0, 1]))
        X_temp['y'] = y
        # Drop duplicates:
        X_temp = X_temp.drop_duplicates()
        y = X_temp[['y']]

        num_zeros = X_temp[X_temp['y'] == 0].to_numpy().shape[0]
        num_ones = X_temp[X_temp['y'] == 1].to_numpy().shape[0]

        for mode, clf in product(MODES, classifiers):
            # print(mode)
            dict_metrics, num_folds = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data=mode,
                                                     num_folds=INITIAL_FOLDS,
                                                     n_neighbours=N_NEIGH, clf=clf, k=k, theta=theta)
            dict_metrics = add_metainfo_dataset(dict_metrics, dataset, num_ones, num_zeros, mode, N_NEIGH, clf)
            df_result = df_result.append(dict_metrics, ignore_index=True)

        # dict_metrics = {k: dict_metrics_1.get(k, 0) + dict_metrics_2.get(k, 0) + dict_metrics_3.get(k, 0) for k in
        #                set(dict_metrics_1) | set(dict_metrics_2) | set(dict_metrics_3)}

        # assert df_result.shape[1] == len(dict_metrics.keys()) or \
        #       (dict_metrics['NUM_fails'] == dict_metrics['NUM_fails_gamma'])
        # assert set(df_result.columns) == set(dict_metrics.keys()) or \
        #       (dict_metrics['NUM_fails'] == dict_metrics['NUM_fails_gamma'])

    # print(df_result)
    success = get_number_success(df_result)
    df_result = add_metadata(df_result, k, theta, success)
    print('Saving output_{}_{}_{}.xlsx'.format(k, theta, success))
    df_result.to_excel("output_{}_{}_{}.xlsx".format(k, theta, success), index=False)
