#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 00:25:51 2017

@author: Rodrigo Azevedo
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing train dataset
dataset = pd.read_table('data.txt')
X = dataset.values[:,:3]
r_y = dataset.values[:,3]
y = [1.0 if i == 1 else 0.0 for i in r_y]
y = np.array(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Loading libraries
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Logistic Regression
accuracies = []
log_losses = []
precision_1 = []
precision_0 = []

for i in range(10):
    skf = StratifiedKFold(n_splits = 10, shuffle=True)
    for train, test in skf.split(X, y):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        classifier = LogisticRegression(C=1/2)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_prob = classifier.predict_proba(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        precision_1.append(precision_score(y_test, y_pred, pos_label=1))
        precision_0.append(precision_score(y_test, y_pred, pos_label=0))
        log_losses.append(log_loss(y_test, y_prob))

print('10 x stratified 10-fold cross-validation')
print('Acurácia: '+ str((np.array(accuracies)).mean()) + ' +/- ' + str((np.array(accuracies)).std()))
print('Log Loss: ' + str((np.array(log_losses)).mean()) + ' +/- ' + str((np.array(log_losses)).std()))
print('Precision (+1): ' + str((np.array(precision_1)).mean()) + ' +/- ' + str((np.array(precision_1)).std()))
print('Precision (-1): ' + str((np.array(precision_0)).mean()) + ' +/- ' + str((np.array(precision_0)).std()))
print('\n')

# Training
classifier = LogisticRegression(C=1/2)
classifier.fit(X, y)

# Classifying train set
y_pred = classifier.predict(X)
y_prob = classifier.predict_proba(X)
accuracy = accuracy_score(y, y_pred)
precision_1 = precision_score(y, y_pred, pos_label=1)
precision_0 = precision_score(y, y_pred, pos_label=0)
log_l  = log_loss(y, y_prob)
cm = confusion_matrix(y, y_pred)

print('Valores para predição no conjunto de treino')
print('Acurácia: '+ str(accuracy))
print('Log Loss: ' + str(log_l))
print('Precision (+1): ' + str(precision_1))
print('Precision (-1): ' + str(precision_0))
print('Matriz de confusão: ')
print(cm)
print('\n')

print('Importando test set')
# Importing test dataset
dataset = pd.read_table('test.txt')
X = dataset.values[:,:3]
r_y = dataset.values[:,3]
y = [1.0 if i == 1 else 0.0 for i in r_y]
y = np.array(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Classifying train set
y_pred = classifier.predict(X)
y_prob = classifier.predict_proba(X)
accuracy = accuracy_score(y, y_pred)
precision_1 = precision_score(y, y_pred, pos_label=1)
precision_0 = precision_score(y, y_pred, pos_label=0)
log_l = log_loss(y, y_prob)
cm = confusion_matrix(y, y_pred)

print('Valores para predição no conjunto de teste')
print('Acurácia: '+ str(accuracy))
print('Log Loss: ' + str(log_l))
print('Precision (+1): ' + str(precision_1))
print('Precision (-1): ' + str(precision_0))
print('Matriz de confusão: ')
print(cm)
print('\n')