from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def cv_metrics(X, y, model, num_runs=3):
    gmean_total = np.zeros(6)
    recall_total = np.zeros(6)
    accuracy_total = np.zeros(6)
    f1score_total = np.zeros(6)
    for i in range(1, num_runs):
        accuracy, recall, gmean, f1score = model_metrics(X, y, i, model)
        accuracy_total = accuracy_total + accuracy
        recall_total = recall_total + recall
        gmean_total = gmean_total + gmean
        f1score_total = f1score_total + f1score

        Totals = np.c_[accuracy_total, recall_total, gmean_total, f1score_total] / (num_runs - 1)

    return pd.DataFrame(data=Totals, columns=['Accuracy', 'Recall', 'G-mean', 'F1 score'],
                        index=['raw', 'nearmiss', 'random', 'smote', 'ada', 'kde'])


def model_metrics(X, y, i, model):  # i rep random state in the train/test split
    accuracies = []
    recalls = []
    gmeans = []
    f1scores = []
    for metrik in ['smote']:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i, stratify=y, test_size=0.3)
        # X_train, X_test, y_train, y_test = X, X, y, y
        if metrik == 'smote':
            smt = SMOTE(random_state=42)
            X_train, y_train = smt.fit_sample(X_train, y_train)

        clf = train_model(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred = (y_pred > 0.5)

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        gmean = g_mean(confusion_matrix(y_test, y_pred))
        f1score = f1_score(y_test, y_pred)

        accuracies.append(accuracy)
        recalls.append(recall)
        gmeans.append(gmean)
        f1scores.append(f1score)

    return np.array(accuracies), np.array(recalls), np.array(gmeans), np.array(f1scores)


s = 'ecoli'
fetched = fetch_datasets()[s]
X = fetched.data.copy()
y = fetched.target.copy()
y[y == -1] = 0
num_zeros = y[y == 0].shape[0]
num_ones = y[y == 1].shape[0]

# When str, specify the class targeted by the resampling. The number of samples in the different classes will be equalized. Possible choices are
# 'not majority': resample all classes but the majority class;
#
# Explanation: https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html#smote-adasyn
smt_1 = SMOTE(sampling_strategy='auto', k_neighbors=5)  # random_state=42

print('X initail shape:', X.shape)
print('y initial shape:', y.shape)
print('y_mijority initial:', num_zeros)
print('y_minority initial:', num_ones)
# X_train, y_train = smt_1.fit_sample(X, y) # fit_sample is alias to fit_resample
X_train, y_train = smt_1.fit_resample(X, y)
num_zeros = y_train[y_train == 0].shape[0]
num_ones = y_train[y_train == 1].shape[0]
print('X SMOTE shape:', X_train.shape)
print('y SMOTE shape:', y_train.shape)
print('y SMOTE_minority initial:', num_zeros)
print('y SMOTE_majority initial:', num_ones)

summary = cv_metrics(X, y, model='nn')
