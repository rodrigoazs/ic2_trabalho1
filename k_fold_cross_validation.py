# k-Fold Cross Validation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_table('data.txt')
X = dataset.values[:,:3]
r_y = dataset.values[:,3]
y = [1.0 if i == 1 else 0.0 for i in r_y]
y = np.array(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Oversampling
ros = RandomOverSampler()
X, y = ros.fit_sample(X, y)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score

# Applying k-Fold Cross Validation
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut

# Logistic Regression
accuracies = []
log_losses = []
precision_1 = []
precision_0 = []

for i in range(10):
    skf = StratifiedKFold(n_splits = 10, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        precision_1.append(precision_score(y_test, y_pred, pos_label=1))
        precision_0.append(precision_score(y_test, y_pred, pos_label=0))
        log_losses.append(log_loss(y_test, y_prob))

print('Accuracy')
print('Mean:' + str((np.array(accuracies)).mean()))
print('Std:' + str((np.array(accuracies)).std()))
print('Precision 1')
print('Mean:' + str((np.array(precision_1)).mean()))
print('Std:' + str((np.array(precision_1)).std()))
print('Precision 0')
print('Mean:' + str((np.array(precision_0)).mean()))
print('Std:' + str((np.array(precision_0)).std()))
print('Log Loss')
print('Mean:' + str((np.array(log_losses)).mean()))
print('Std:' + str((np.array(log_losses)).std()))

#accuracies = []     
#skf = LeaveOneOut()
#for train, test in skf.split(X):
#    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
#    classifier = LogisticRegression()
#    classifier.fit(X_train, y_train)
#    y_pred = classifier.predict(X_test)
#    accuracies.append(accuracy_score(y_test, y_pred))

# MLP
accuracies = []
precision_1 = []
precision_0 = []
log_losses = []

for i in range(10):
    skf = StratifiedKFold(n_splits = 10, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        classifier = MLPClassifier(hidden_layer_sizes=(6,))
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        precision_1.append(precision_score(y_test, y_pred, pos_label=1))
        precision_0.append(precision_score(y_test, y_pred, pos_label=0))
        log_losses.append(log_loss(y_test, y_prob))

print('Accuracy')
print('Mean:' + str((np.array(accuracies)).mean()))
print('Std:' + str((np.array(accuracies)).std()))
print('Precision 1')
print('Mean:' + str((np.array(precision_1)).mean()))
print('Std:' + str((np.array(precision_1)).std()))
print('Precision 0')
print('Mean:' + str((np.array(precision_0)).mean()))
print('Std:' + str((np.array(precision_0)).std()))
print('Log Loss')
print('Mean:' + str((np.array(log_losses)).mean()))
print('Std:' + str((np.array(log_losses)).std()))

# RandomForest
accuracies = []
precision_1 = []
precision_0 = []
log_losses = []

for i in range(10):
    skf = StratifiedKFold(n_splits = 10, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        precision_1.append(precision_score(y_test, y_pred, pos_label=1))
        precision_0.append(precision_score(y_test, y_pred, pos_label=0))
        log_losses.append(log_loss(y_test, y_prob))

print('Accuracy')
print('Mean:' + str((np.array(accuracies)).mean()))
print('Std:' + str((np.array(accuracies)).std()))
print('Precision 1')
print('Mean:' + str((np.array(precision_1)).mean()))
print('Std:' + str((np.array(precision_1)).std()))
print('Precision 0')
print('Mean:' + str((np.array(precision_0)).mean()))
print('Std:' + str((np.array(precision_0)).std()))
print('Log Loss')
print('Mean:' + str((np.array(log_losses)).mean()))
print('Std:' + str((np.array(log_losses)).std()))

# kNN
accuracies = []
precision_1 = []
precision_0 = []
log_losses = []

for i in range(10):
    skf = StratifiedKFold(n_splits = 10, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        precision_1.append(precision_score(y_test, y_pred, pos_label=1))
        precision_0.append(precision_score(y_test, y_pred, pos_label=0))
        log_losses.append(log_loss(y_test, y_prob))

print('Accuracy')
print('Mean:' + str((np.array(accuracies)).mean()))
print('Std:' + str((np.array(accuracies)).std()))
print('Precision 1')
print('Mean:' + str((np.array(precision_1)).mean()))
print('Std:' + str((np.array(precision_1)).std()))
print('Precision 0')
print('Mean:' + str((np.array(precision_0)).mean()))
print('Std:' + str((np.array(precision_0)).std()))
print('Log Loss')
print('Mean:' + str((np.array(log_losses)).mean()))
print('Std:' + str((np.array(log_losses)).std()))