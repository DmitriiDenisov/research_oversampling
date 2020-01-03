from utils.utils import get_dataset_pd, aug_train, get_metrics

import numpy as np

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier


def handle_dataset(name_dataset: str, dict_metrics: dict, aug_data: bool, num_folds=5):
    """

    :param name_dataset: str
    :param dict_metrics:
    :param aug_data:
    :param num_folds:
    :return:
    """
    if not aug_data:
        # 1. Get dataset:
        X, y = get_dataset_pd(name_dataset)
        kf = KFold(n_splits=num_folds)
        kf.get_n_splits(X)

        for train_index, test_index in kf.split(X):
            # 2. Split on test and train
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if y_train[0].sum() < 2 or y_test[0].sum() < 2:
                num_folds -= 1
                continue

            # 3. Fit
            clf = RandomForestClassifier(n_estimators=50)
            clf.fit(X_train, y_train.to_numpy().flatten())

            # 4. Predict and metrics:
            y_pred = clf.predict(X_test)
            dict_temp = get_metrics(y_test, y_pred, aug_data)
            dict_metrics = {k: dict_metrics.get(k, 0) + dict_temp.get(k, 0) for k in set(dict_metrics) | set(dict_temp)}
        if num_folds:  # in case every k-fold does not have at least two minority points
            # metrics = metrics / num_folds
            dict_metrics = {k: v / num_folds for k, v in dict_metrics.items()}
        # dict(list(dict_metrics.items()) + list(dict('pr').items()))
    else:
        # 1. Get dataset:
        X, y = get_dataset_pd(name_dataset)
        X['y'] = y
        kf = KFold(n_splits=num_folds)
        kf.get_n_splits(X)

        for train_index, test_index in kf.split(X):
            # 2. Split on test and train
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if y_train[0].sum() < 2 or y_test[0].sum() < 2:
                num_folds -= 1
                continue
            X_test = X_test.drop('y', 1)

            # 3. Augment train part by generating new minority points
            X_train_aug = aug_train(X_train)

            # 4. Shuffle
            X_train_aug = X_train_aug.sample(frac=1)  # shuffle

            # 5. Drop target from train
            y_train_aug = X_train_aug['y']
            X_train_aug = X_train_aug.drop('y', 1)

            # 6. Fit:
            clf = RandomForestClassifier(n_estimators=50)
            clf.fit(X_train_aug, y_train_aug.to_numpy().flatten())

            # 7. Predict and metrics:
            y_pred = clf.predict(X_test)
            dict_temp = get_metrics(y_test, y_pred, aug_data)
            dict_metrics = {k: dict_metrics.get(k, 0) + dict_temp.get(k, 0) for k in set(dict_metrics) | set(dict_temp)}
        if num_folds:  # in case every k-fold does not have at least two minority points
            dict_metrics = {k: v / num_folds for k, v in dict_metrics.items()}
    return dict_metrics, num_folds
