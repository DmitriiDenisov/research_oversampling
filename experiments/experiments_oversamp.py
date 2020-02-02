from imblearn.datasets import fetch_datasets
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
overs = RandomOverSampler(random_state=42)

print('X initail shape:', X.shape)
print('y initial shape:', y.shape)
print('y_mijority initial:', num_zeros)
print('y_minority initial:', num_ones)
# X_train, y_train = smt_1.fit_sample(X, y) # fit_sample is alias to fit_resample
X_train, y_train = overs.fit_resample(X, y)
num_zeros = y_train[y_train == 0].shape[0]
num_ones = y_train[y_train == 1].shape[0]
print('X OVERSAMP shape:', X_train.shape)
print('y OVERSAMP shape:', y_train.shape)
print('y OVERSAMP_minority new:', num_zeros)
print('y OVERSAMP_majority new:', num_ones)
