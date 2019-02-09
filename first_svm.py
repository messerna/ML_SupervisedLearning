#Importing required libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


from sklearn.metrics import classification_report,confusion_matrix

from sklearn import svm
#clf = svm.SVC(gamma=0.001, C=100.)
clf = svm.SVC(kernel = 'linear')
#clf = svm.SVC(kernel = 'poly', degree=3)
#clf = svm.SVC(kernel = 'rbf')    #gaussian
#clf = svm.SVC(kernel = 'sigmoid')

data = pd.read_csv('parkinsons.csv')
#print(data.head)
y = data['status']
X = data.drop(columns=['status','name'])

#Using the train_test_split to create train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 47, test_size = 0.25)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf.fit(X_train, y_train)

#Predicting labels on the test set.
y_pred =  clf.predict(X_test)

#Importing the accuracy metric from sklearn.metrics library

from sklearn.metrics import accuracy_score
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 

w = clf.coef_[0]
print(w)

a = -w[0] / w[1]

xx = np.linspace(0,12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.legend()
plt.show()