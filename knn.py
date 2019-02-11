'''
# boosting.py
# 
# Runs knn model with 1 and 3 nearest neighbors 
#
# Neal Messer
# nmesser3@gatech.edu
# 
# Skeleton of code and basic information on k-nearest neighbors taken from sklearn documentation at:
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
#
# Code modified for GA Tech OMSCS 7641 Machine Learning Class
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
	print ('\nChoose dataset: income OR personality\nUsage: knn.py income OR knn.py personality')
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

for i in range(1,4,2):
	knn = KNeighborsClassifier(n_neighbors=i)
	graph_learning_curve(knn, '{0} {1} neighbor knn learning curve'.format(title, i), X, y )

	# start training clock
	training_start_time = time.perf_counter()

	knn.fit(X_train, y_train)

	# start prediction clock
	prediction_start_time = time.perf_counter()

	#Predicting labels on the test set.
	y_pred =  knn.predict(X_test)

	prediction_stop_time = time.perf_counter()

	print('K-Nearest Neighbors: n = {0}'.format(i))
	print('total training time was %.6f' % (prediction_start_time - training_start_time))
	print('total prediction time was %.6f' % (prediction_stop_time - prediction_start_time))

	y_pred_train=knn.predict(X_train)
	print('Accuracy Score on train data: %.4f' % (accuracy_score(y_true=y_train, y_pred=y_pred_train)))
	print('Accuracy Score on test data: %.4f' % (accuracy_score(y_true=y_test, y_pred=y_pred)))
	print('F1 Score Score on train data: %.4f' % (f1_score(y_true=y_train, y_pred=y_pred_train)))
	print('F1 Score Score on test data: %.4f' % (f1_score(y_true=y_test, y_pred=y_pred))) 

	out_file = "knn_%d_%d.csv" % (i, time.time())
	f = open(out_file, "a")
	f.write("%.4f,%.4f,%.4f,%.4f,%.6f,%.6f\n" % ((accuracy_score(y_true=y_train, y_pred=y_pred_train)), (accuracy_score(y_true=y_test, y_pred=y_pred)), (f1_score(y_true=y_train, y_pred=y_pred_train)), (f1_score(y_true=y_test, y_pred=y_pred)), (prediction_start_time - training_start_time), (prediction_stop_time - prediction_start_time)))
	
plt.show()