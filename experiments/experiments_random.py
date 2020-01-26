import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from experiments.functions_for_random import get_random_gamma
from utils.utils import get_dataset_pd

np.random.seed(46)

X_temp, y = get_dataset_pd('optical_digits')
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_temp, y)
y_pred = classifier.predict_proba(X_temp)

print(y_pred[:10])

s = get_random_gamma()
print(s)

np.random.seed(456)

s = get_random_gamma()
print(s)