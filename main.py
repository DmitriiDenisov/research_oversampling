from utils.handle_dataset import handle_dataset
from os import path
from tqdm import tqdm
import pandas as pd
import numpy as np

from utils.utils import get_dataset_pd, add_metainfo_dataset

datasets = ['ecoli',
            'optical_digits',
            'satimage',
            'pen_digits',
            'abalone',
            'sick_euthyroid',
            'spectrometer',
            'car_eval_34',
            # 'isolet', # около 10 минут считается
            'us_crime',
            'yeast_ml8',
            'scene',
            'libras_move',
            'thyroid_sick',
            'coil_2000',
            'arrhythmia',
            'solar_flare_m0',
            'oil',
            'car_eval_4',
            'wine_quality',
            'letter_img',
            'yeast_me2',
            'webpage',
            # 'ozone_level',
            # 'mammography', # долгий!!!!
            # 'protein_homo',
            'abalone_19']

#datasets = [
#    'isolet', ]

if path.exists('output.xlsx'):
    df_result = pd.read_excel('output.xlsx', index_col=None)
else:
    df_result = pd.DataFrame(
        columns=['NAME_Dataset', 'NUM_elements', 'minority_perc', 'Generated_points', 'NUM_fails', 'f1_score',
                 'precision',
                 'recall', 'AUC_PR'])
INITIAL_FOLDS = 5
N_NEIGH = 5

for dataset in tqdm(datasets):
    X_temp, y = get_dataset_pd(dataset)
    assert np.all(np.unique(y) == np.array([0, 1]))
    X_temp['y'] = y
    # Drop duplicates:
    X_temp = X_temp.drop_duplicates()
    y = X_temp[['y']]

    num_zeros = X_temp[X_temp['y'] == 0].to_numpy().shape[0]
    num_ones = X_temp[X_temp['y'] == 1].to_numpy().shape[0]

    for mode in ['initial', 'gamma', 'smote']:
        print(mode)
        dict_metrics, num_folds = handle_dataset(X_temp.drop('y', 1), y, dict(), aug_data=mode, num_folds=INITIAL_FOLDS,
                                                 n_neighbours=N_NEIGH)
        dict_metrics = add_metainfo_dataset(dict_metrics, dataset, num_ones, num_zeros, mode)
        df_result = df_result.append(dict_metrics, ignore_index=True)

    # dict_metrics = {k: dict_metrics_1.get(k, 0) + dict_metrics_2.get(k, 0) + dict_metrics_3.get(k, 0) for k in
    #                set(dict_metrics_1) | set(dict_metrics_2) | set(dict_metrics_3)}

    # assert df_result.shape[1] == len(dict_metrics.keys()) or \
    #       (dict_metrics['NUM_fails'] == dict_metrics['NUM_fails_gamma'])
    # assert set(df_result.columns) == set(dict_metrics.keys()) or \
    #       (dict_metrics['NUM_fails'] == dict_metrics['NUM_fails_gamma'])

# print(df_result)
df_result.to_excel("output.xlsx", index=False)
