import imblearn

from utils.utils import get_dataset_pd, aug_train

print(imblearn.__version__)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# 1. Get dataset:
X, y = get_dataset_pd('abalone')

# 2. Split on test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # random_state=42

# 3. Fit:
clf = RandomForestClassifier(n_estimators=50)
clf.fit(X_train, y_train.to_numpy().flatten())

# 4. Predict and metrics:
y_pred = clf.predict(X_test)
print(f1_score(y_test.to_numpy().flatten(), y_pred))
print(precision_score(y_test.to_numpy().flatten(), y_pred))
print(recall_score(y_test.to_numpy().flatten(), y_pred))

# ------------------------------------

# 1. Get dataset:
X, y = get_dataset_pd('abalone')

X['y'] = y

# 2. Split on test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # random_state=42
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
print(f1_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))