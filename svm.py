'''
# boosting.py
# 
# Runs svm model with linear and 3rd degree poly kernels 
#
# Neal Messer
# nmesser3@gatech.edu
# 
# Skeleton of code and basic information on svm from:
# https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html
#
# Code modified for GA Tech OMSCS 7641 Machine Learning Class
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, sys
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from graph_learning_curve import graph_learning_curve

# parse args
if ( sys.argv[1] == 'income'):
	input_file = "datasets/income.csv"
	output_col = "Column 15"
	title = "Income"
elif ( sys.argv[1] == 'personality'):
	input_file = "datasets/personality_cleaned.csv"
	output_col = "nerdy"
	title = "Personality"
else:
	print ('\nChoose dataset: income OR personality\nUsage: svm.py income OR svm.py personality')
	exit(1)


# read in training file
data = pd.read_csv(input_file)

# set up training data
y = data[output_col]       
X = data.drop(columns=[output_col])

# use the train_test_split to create train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 12, test_size = 1/3)

print('Number of examples in the train data:', X_train.shape[0])
print('Number of examples in the test data:', X_test.shape[0])

# scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

#clf = SVC(gamma=0.001, C=100.)
clf = SVC(kernel = 'linear')
#clf = SVC(kernel = 'poly', degree=3)
#clf = SVC(kernel = 'rbf')    #gaussian
#clf = SVC(kernel = 'sigmoid')
graph_learning_curve(clf, '{0} linear svm learning curve'.format(title), X, y )

# start training clock
training_start_time = time.perf_counter()

# train model
clf.fit(X_train, y_train)

# start prediction clock
prediction_start_time = time.perf_counter()

#Predicting labels on the test set.
y_pred =  clf.predict(X_test)

prediction_stop_time = time.perf_counter()

print('SVM Linear')
print('total training time was %.6f' % (prediction_start_time - training_start_time))
print('total prediction time was %.6f' % (prediction_stop_time - prediction_start_time))

y_pred_train=clf.predict(X_train)
print('Accuracy Score on train data: %.4f' % (accuracy_score(y_true=y_train, y_pred=y_pred_train)))
print('Accuracy Score on test data: %.4f' % (accuracy_score(y_true=y_test, y_pred=y_pred)))
print('F1 Score Score on train data: %.4f' % (f1_score(y_true=y_train, y_pred=y_pred_train)))
print('F1 Score Score on test data: %.4f' % (f1_score(y_true=y_test, y_pred=y_pred))) 

out_file = "svml%d.csv" % time.time()
f = open(out_file, "a")
f.write("%.4f,%.4f,%.4f,%.4f,%.6f,%.6f\n" % ((accuracy_score(y_true=y_train, y_pred=y_pred_train)), (accuracy_score(y_true=y_test, y_pred=y_pred)), (f1_score(y_true=y_train, y_pred=y_pred_train)), (f1_score(y_true=y_test, y_pred=y_pred)), (prediction_start_time - training_start_time), (prediction_stop_time - prediction_start_time)))


#clf = SVC(gamma=0.001, C=100.)
#clf = SVC(kernel = 'linear')
clf = SVC(kernel = 'poly', degree=3)
#clf = SVC(kernel = 'rbf')    #gaussian
#clf = SVC(kernel = 'sigmoid')
graph_learning_curve(clf, '{0} 3rd degree polynomial svm learning curve'.format(title), X, y )

# start training clock
training_start_time = time.perf_counter()

# train model
clf.fit(X_train, y_train)

# start prediction clock
prediction_start_time = time.perf_counter()

#Predicting labels on the test set.
y_pred =  clf.predict(X_test)

prediction_stop_time = time.perf_counter()

print('\n\nSVM 3rd Degree Polynomial')
print('total training time was %.6f' % (prediction_start_time - training_start_time))
print('total prediction time was %.6f' % (prediction_stop_time - prediction_start_time))

y_pred_train=clf.predict(X_train)
print('Accuracy Score on train data: %.4f' % (accuracy_score(y_true=y_train, y_pred=y_pred_train)))
print('Accuracy Score on test data: %.4f' % (accuracy_score(y_true=y_test, y_pred=y_pred)))
print('F1 Score Score on train data: %.4f' % (f1_score(y_true=y_train, y_pred=y_pred_train)))
print('F1 Score Score on test data: %.4f' % (f1_score(y_true=y_test, y_pred=y_pred))) 

out_file = "svm3dp%d.csv" % time.time()
f = open(out_file, "a")
f.write("%.4f,%.4f,%.4f,%.4f,%.6f,%.6f\n" % ((accuracy_score(y_true=y_train, y_pred=y_pred_train)), (accuracy_score(y_true=y_test, y_pred=y_pred)), (f1_score(y_true=y_train, y_pred=y_pred_train)), (f1_score(y_true=y_test, y_pred=y_pred)), (prediction_start_time - training_start_time), (prediction_stop_time - prediction_start_time)))

plt.show()