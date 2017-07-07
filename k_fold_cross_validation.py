# k-Fold Cross Validation

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_table('data.txt')
X = dataset.values[:,:3]
y = dataset.values[:,3]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', probability=True)
#classifier.fit(X_train, y)

# Predicting the Test set results
#y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, scoring='neg_log_loss', X = X_train, y = y, cv = 10)
m = accuracies.mean()
s = accuracies.std()

classifier.fit(X_train, y)