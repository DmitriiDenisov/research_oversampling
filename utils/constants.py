INITIAL_FOLDS = 5
N_NEIGH = 3
K = 1 / 8
THETA = 2.
MODES = ['initial', 'gamma', 'smote']
COLUMNS = ['NAME_Dataset', 'Algo', 'N_neigh', 'NUM_elements',
           'minority_perc', 'Generated_points', 'NUM_fails', 'f1_score',
           'precision', 'recall', 'AUC_PR']

DATASETS = ['ecoli',
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
            # 'webpage', # долгий, shape=(34780, 300)
            'ozone_level',
            'mammography',
            # 'protein_homo', # долгий, shape=(144968, 75)!!!!
            'abalone_19']