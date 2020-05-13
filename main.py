# for i in {1..10}; do python3 main.py --seed $i; done
# for i in {1..10}; do python3 main.py --seed $i; done
import datetime
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', dest='seed', type=int, default=1,
                    help='random seed')
args = parser.parse_args()

# Seed value
# Apparently you may use different seed values at each stage
seed_value = args.seed

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
# from tensorflow import set_random_seed

# set_random_seed(seed_value)
import tensorflow

tensorflow.random.set_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K

# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True,
#                              device_count={'CPU': 1})
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)

# удивительно, но это надо для того, чтобы зафиксировать рандом в Керасе
# Пруф: https://github.com/keras-team/keras/issues/2743
from keras.models import Sequential

import os
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from utils.handle_dataset import handle_dataset
from utils.constants import *

from utils.utils import get_dataset_pd, add_metainfo_dataset, get_NN, add_metadata, get_number_success

print(f'Started: {datetime.datetime.now()}')

# DATASETS = [
#    'spectrometer'
# ]
MODES = ['smote+normal']
# MODES = ['initial', 'smote']
# DATASETS = ['pen_digits']
# MODES = ['smote+normal']
# DATASETS = ['synthetic']
# DATASETS = ['ecoli',
#            'optical_digits',
#            'satimage']

# list_k_theta = [[0.125, 2.], [1.5, 6.5], [1.7, 7], [1.7, 2], [1, 2], [1, 4]]
# list_k_theta = [[1, 2], [1, 2.5]]
list_k_theta = [[1, 2]]

# INITIAL_FOLDS = 10
# print(INITIAL_FOLDS)


print('Random_seed:', seed_value)
for (k, theta) in list_k_theta:
    print(k, theta)
    if path.exists('output.xlsx'):
        df_result = pd.read_excel('output.xlsx', index_col=None)
    else:
        df_result = pd.DataFrame(
            columns=COLUMNS)

    for dataset in DATASETS:  # !!!!!!!!!!!!!!!!
        # continue # !!!!!
        print(dataset)

        X_temp, y = get_dataset_pd(dataset)
        classifiers = [get_NN(X_temp), RandomForestClassifier(n_estimators=50), DecisionTreeClassifier(),
                       SVC(gamma='auto')]
        # classifiers = [DecisionTreeClassifier()]
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
    success = get_number_success(df_result, index=len(classifiers), step=4, num_modes=len(MODES))
    df_result = add_metadata(df_result, k, theta, success, seed_value)
    print(f'Saving output_{k}_{theta}_success_{success}_seed_{seed_value}.xlsx')
    #df_result.to_excel(f"compare_temp/output_{k}_{theta}_success_{success}_seed_{seed_value}.xlsx",
    #                    index=False)

    i = 0
    os.makedirs("compare_temp", exist_ok=True)
    while os.path.isfile(f"compare_temp/{i}.xlsx"):
        i += 1
    df_result.to_excel(f"compare_temp/{i}.xlsx",
                       index=False)

print(f'Finished: {datetime.datetime.now()}')
