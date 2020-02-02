from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from keras.engine.sequential import Sequential
from sklearn.neighbors import NearestNeighbors
import numpy as np

from utils.utils import get_dataset_pd, aug_train, get_metrics
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


def handle_dataset(X: np.array,
                   y: np.array,
                   dict_metrics: dict,
                   aug_data: str,
                   clf,
                   k=1 / 8,
                   theta=2.,
                   num_folds=5,
                   n_neighbours=5,
                   ) -> object:
    """

    :param clf:
    :param k:
    :param theta:
    :param X:
    :param y:
    :param dict_metrics:
    :param aug_data:
    :param num_folds:
    :param n_neighbours:
    :return:
    """
    initial_folds_num = num_folds
    if aug_data == 'initial':
        # 1. Get dataset:
        # X, y = get_dataset_pd(name_dataset)
        kf = KFold(n_splits=num_folds, shuffle=True)
        kf.get_n_splits(X)

        for train_index, test_index in kf.split(X):
            # 2. Split on test and train
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            assert y_train['y'].sum() + y_test['y'].sum() == y['y'].sum()

            if y_train['y'].sum() < 2 or y_test['y'].sum() < 2:
                num_folds -= 1
                continue

            assert y_train['y'].sum() >= n_neighbours

            # 3. Shuffle
            assert len(X_train) == len(y_train)
            p = np.random.permutation(len(X_train))
            X_train = X_train.iloc[p]
            y_train = y_train.iloc[p]

            # 4. Fit
            # clf = RandomForestClassifier(n_estimators=50)
            if isinstance(clf, Sequential):
                clf.fit(X_train, y_train.to_numpy().flatten(), epochs=1, verbose=0, shuffle=False)
            else:
                clf.fit(X_train, y_train.to_numpy().flatten())

            # 5. Predict and metrics:
            y_pred = clf.predict(X_test)
            y_pred = np.around(y_pred)  # для нейронки

            dict_temp = get_metrics(y_test, y_pred, aug_data)
            dict_metrics = {k: dict_metrics.get(k, 0) + dict_temp.get(k, 0) for k in set(dict_metrics) | set(dict_temp)}
        if num_folds:  # in case every k-fold does not have at least two minority points
            # metrics = metrics / num_folds
            dict_metrics = {k: v / num_folds for k, v in dict_metrics.items()}
        # dict(list(dict_metrics.items()) + list(dict('pr').items()))
        dict_metrics['NUM_fails'] = initial_folds_num - num_folds

    elif aug_data == 'gamma':
        # 1. Get dataset:
        # X, y = get_dataset_pd(name_dataset)
        X['y'] = y
        kf = KFold(n_splits=num_folds, shuffle=True)
        kf.get_n_splits(X)

        for train_index, test_index in kf.split(X):
            # 2. Split on test and train
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if y_train['y'].sum() < 2 or y_test['y'].sum() < 2:
                num_folds -= 1
                continue
            X_test = X_test.drop('y', 1)

            # 3. Augment train part by generating new minority points
            X_train_aug = aug_train(X_train, n_neighbours, k, theta)

            # 4. Shuffle
            X_train_aug = X_train_aug.sample(frac=1)  # shuffle

            # 5. Drop target from train
            y_train_aug = X_train_aug['y']
            X_train_aug = X_train_aug.drop('y', 1)

            # 6. Fit:
            # clf = RandomForestClassifier(n_estimators=50)
            if isinstance(clf, Sequential):
                clf.fit(X_train_aug, y_train_aug.to_numpy().flatten(), epochs=10, verbose=0)
            else:
                clf.fit(X_train_aug, y_train_aug.to_numpy().flatten())

            # 7. Predict and metrics:
            y_pred = clf.predict(X_test)
            y_pred = np.around(y_pred)  # для нейронки

            dict_temp = get_metrics(y_test, y_pred, aug_data)
            dict_metrics = {k: dict_metrics.get(k, 0) + dict_temp.get(k, 0) for k in set(dict_metrics) | set(dict_temp)}
        if num_folds:  # in case every k-fold does not have at least two minority points
            dict_metrics = {k: v / num_folds for k, v in dict_metrics.items()}
        dict_metrics['NUM_fails'] = initial_folds_num - num_folds

    elif aug_data == 'smote' or aug_data == 'UNDERSAMP' or aug_data == 'OVERSAMP' or aug_data == 'ADASYN':
        kf = KFold(n_splits=num_folds, shuffle=True)
        kf.get_n_splits(X)

        for train_index, test_index in kf.split(X):
            # 2. Split on test and train
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if y_train['y'].sum() < 2 or y_test['y'].sum() < 2:
                num_folds -= 1
                continue
            X_test = X_test.drop('y', 1, errors='ignore')  # ????

            # 3. Augment train part by generating new minority points
            if aug_data == 'smote':
                algo = SMOTE(sampling_strategy='auto', k_neighbors=5)  # random_state=42
            elif aug_data == 'UNDERSAMP':
                algo = RandomUnderSampler(random_state=42)
            elif aug_data == 'OVERSAMP':
                algo = RandomOverSampler(random_state=42)
            elif aug_data == 'ADASYN':
                algo = ADASYN(random_state=42)

            X_train_aug, y_train_aug = algo.fit_resample(X_train.to_numpy(), y_train.to_numpy().flatten())
            # assert (y_train_aug == 0).sum() == (y_train_aug == 1).sum() == (y_train['y'] == 0).sum()

            # 4. Shuffle
            assert len(X_train_aug) == len(y_train_aug)
            p = np.random.permutation(len(X_train_aug))
            X_train_aug = X_train_aug[p]
            y_train_aug = y_train_aug[p]

            # 5. Fit:
            # clf = RandomForestClassifier(n_estimators=50)
            if isinstance(clf, Sequential):
                clf.fit(X_train_aug, y_train_aug.flatten(), epochs=10, verbose=0)
            else:
                clf.fit(X_train_aug, y_train_aug.flatten())

            # 6. Predict and metrics:
            y_pred = clf.predict(X_test)
            y_pred = np.around(y_pred)  # для нейронки

            dict_temp = get_metrics(y_test, y_pred, aug_data)
            dict_metrics = {k: dict_metrics.get(k, 0) + dict_temp.get(k, 0) for k in set(dict_metrics) | set(dict_temp)}
        if num_folds:  # in case every k-fold does not have at least two minority points
            dict_metrics = {k: v / num_folds for k, v in dict_metrics.items()}
        dict_metrics['NUM_fails'] = initial_folds_num - num_folds

    return dict_metrics, num_folds
