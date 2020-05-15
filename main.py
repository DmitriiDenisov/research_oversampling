# for i in {1..10}; do python3 main.py --seed $i; done
# for i in {1..10}; do python3 main.py --seed $i; done
import datetime
import argparse
import os
import warnings
from typing import List

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)

from itertools import product
from os import path
import pandas as pd
import numpy as np
from utils.handle_dataset import handle_dataset
from utils.constants import *

from utils.utils_py import get_dataset_pd, add_metainfo_dataset, get_NN, add_metadata, get_number_success, get_clf

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', dest='seed', type=int, default=1,
                    help='random seed')
args = parser.parse_args()
# Fix Randomness
seed_value = args.seed

np.random.seed(seed_value)

print(f'Started: {datetime.datetime.now()}')

DATASETS = [
    'spectrometer'
]
INITIAL_FOLDS: int = 3
# MODES = ['smote+normal']
# MODES = ['initial', 'smote']
# DATASETS = ['pen_digits']
# MODES = ['smote+normal']
# DATASETS = ['synthetic']
# DATASETS = ['ecoli',
#            'optical_digits',
#            'satimage']

# list_k_theta = [[0.125, 2.], [1.5, 6.5], [1.7, 7], [1.7, 2], [1, 2], [1, 4]]
# list_k_theta = [[1, 2], [1, 2.5]]
list_k_theta: List[list] = [[1, 2]]
list_k_theta: List[list] = [[2, 0.1], [2, 0.2], [1, 0.2], [3, 0.1], [1, 0.1], [1, 0.3], [1, 2], [3, 0.1]]
# MODES = ['gamma']
# INITIAL_FOLDS = 10
# print(INITIAL_FOLDS)
# MODES = ['initial', 'initial', 'initial', 'initial']

print('Random_seed:', seed_value)
for (k, theta) in list_k_theta:
    print(k, theta)
    if path.exists('output.xlsx'):
        df_result: pd.DataFrame = pd.read_excel('output.xlsx', index_col=None)
    else:
        df_result: pd.DataFrame = pd.DataFrame(
            columns=COLUMNS)

    for dataset in DATASETS:
        print(dataset)

        X_temp, y = get_dataset_pd(dataset)
        # X_temp.sh
        assert np.all(np.unique(y) == np.array([0, 1]))
        X_temp['y'] = y
        X_temp = X_temp.drop_duplicates()
        y = X_temp[['y']]

        num_zeros = X_temp[X_temp['y'] == 0].to_numpy().shape[0]
        num_ones = X_temp[X_temp['y'] == 1].to_numpy().shape[0]

        num_f = X_temp.drop('y', 1).shape[1]
        for mode, clf_str in product(MODES, CLASSIFIERS):
            # print(mode)
            clf = get_clf(clf_str, num_f)
            dict_metrics, num_folds = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data=mode,
                                                     num_folds=INITIAL_FOLDS,
                                                     n_neighbours=N_NEIGH, clf=clf, k=k, theta=theta)
            dict_metrics = add_metainfo_dataset(dict_metrics, dataset, num_ones, num_zeros, mode, N_NEIGH, clf)
            df_result = df_result.append(dict_metrics, ignore_index=True)

    success = get_number_success(df_result, index=len(CLASSIFIERS), step=4, num_modes=len(MODES))
    df_result = add_metadata(df_result, k, theta, success, seed_value)
    print(f'Saving output_{k}_{theta}_success_{success}_seed_{seed_value}.xlsx')
    # df_result.to_excel(f"compare_temp/output_{k}_{theta}_success_{success}_seed_{seed_value}.xlsx",
    #                    index=False)

    i = 0
    os.makedirs("compare_temp", exist_ok=True)
    while os.path.isfile(f"compare_temp/{i}.xlsx"):
        i += 1
    df_result.to_excel(f"compare_temp/{i}.xlsx",
                       index=False)

print(f'Finished: {datetime.datetime.now()}')
