from handle_dataset import handle_dataset
import pandas as pd
import numpy as np

from utils.utils import get_dataset_pd

datasets = ['ecoli', 'satimage', 'abalone', 'spectrometer', 'yeast_ml8', 'scene', 'libras_move', 'wine_quality',
            'letter_img', 'yeast_me2', 'ozone_level', 'mammography']
datasets = ['satimage']

df_result = pd.DataFrame(
    columns=['NAME_Dataset', 'NUM_elements', 'minority_perc', 'Generated_points', 'NUM_fails', 'f1_score', 'precision',
             'recall', 'AUC_PR', 'NUM_fails_gamma', 'f1_score_gamma', 'precision_gamma', 'recall_gamma',
             'AUC_PR_gamma', 'NUM_fails_smote', 'f1_score_smote', 'precision_smote', 'recall_smote',
             'AUC_PR_smote'])
INITIAL_FOLDS = 5
N_NEIGH = 5

for dataset in datasets:
    print(dataset)
    X_temp, y = get_dataset_pd(dataset)
    X_temp['y'] = y
    # Drop duplicates:
    X_temp = X_temp.drop_duplicates()
    y = X_temp[['y']]

    num_zeros = X_temp[X_temp['y'] == 0].to_numpy().shape[0]
    num_ones = X_temp[X_temp['y'] == 1].to_numpy().shape[0]

    dict_metrics_1, num_folds = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='no', num_folds=INITIAL_FOLDS)
    dict_metrics_2, num_folds_gamma = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='gamma',
                                                     num_folds=INITIAL_FOLDS,
                                                     n_neighbours=N_NEIGH)
    dict_metrics_3, num_folds_smote = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data='smote',
                                                     num_folds=INITIAL_FOLDS,
                                                     n_neighbours=N_NEIGH)

    dict_metrics = {k: dict_metrics_1.get(k, 0) + dict_metrics_2.get(k, 0) + dict_metrics_3.get(k, 0) for k in
                    set(dict_metrics_1) | set(dict_metrics_2) | set(dict_metrics_3)}

    dict_metrics['NAME_Dataset'] = dataset
    dict_metrics['NUM_elements'] = X_temp.shape[0]
    dict_metrics['minority_perc'] = num_ones / (num_ones + num_zeros)
    dict_metrics['Generated_points'] = num_zeros - num_ones

    assert df_result.shape[1] == len(dict_metrics.keys()) or \
           (dict_metrics['NUM_fails'] == dict_metrics['NUM_fails_gamma'])
    assert set(df_result.columns) == set(dict_metrics.keys()) or \
           (dict_metrics['NUM_fails'] == dict_metrics['NUM_fails_gamma'])
    df_result = df_result.append(dict_metrics, ignore_index=True)

# print(df_result)
df_result.to_excel("output.xlsx", index=False)
