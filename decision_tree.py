'''
# decision_tree.py
# 
# Runs decision tree model 
#
# Neal Messer
# nmesser3@gatech.edu
# 
# Skeleton of code and basic information on decision trees from:
# https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/ml-decision-tree/tutorial/
#
# Code modified for GA Tech OMSCS 7641 Machine Learning Class
'''
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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
	print ('Choose dataset: income OR personality\nUsage: decision_tree.py income OR decision_tree.py personality')
	exit(1)

# read in training file
data = pd.read_csv(input_file)

# set up training data
y = data[output_col]       
X = data.drop(columns=[output_col])

#Using the train_test_split to create train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 12, test_size = 1/3)

# Decision tree classifier from sklearn 
clf = DecisionTreeClassifier(criterion = 'entropy')

# graph learning curve
graph_learning_curve(clf, '{0} decision tree learning curve'.format(title), X, y )

# start training clock
training_start_time = time.perf_counter()

# Train the decision tree classifier 
clf.fit(X_train, y_train)

# start prediction clock
prediction_start_time = time.perf_counter()

# Predict labels on the test set.
y_pred =  clf.predict(X_test)
prediction_stop_time = time.perf_counter()

# print results

print('Standard Decision Tree')
print('Number of examples in the train data:', X_train.shape[0])
print('Number of examples in the test data:', X_test.shape[0])

print('total training time was %.6f' % (prediction_start_time - training_start_time))
print('total prediction time was %.6f' % (prediction_stop_time - prediction_start_time))

y_pred_train=clf.predict(X_train)
print('Accuracy Score on train data: %.4f' % (accuracy_score(y_true=y_train, y_pred=y_pred_train)))
print('Accuracy Score on test data: %.4f' % (accuracy_score(y_true=y_test, y_pred=y_pred)))
print('F1 Score Score on train data: %.4f' % (f1_score(y_true=y_train, y_pred=y_pred_train)))
print('F1 Score Score on test data: %.4f' % (f1_score(y_true=y_test, y_pred=y_pred))) 

out_file = "dt%d.csv" % time.time()
f = open(out_file, "a")
f.write("%.4f,%.4f,%.4f,%.4f,%.6f,%.6f\n" % ((accuracy_score(y_true=y_train, y_pred=y_pred_train)), (accuracy_score(y_true=y_test, y_pred=y_pred)), (f1_score(y_true=y_train, y_pred=y_pred_train)), (f1_score(y_true=y_test, y_pred=y_pred)), (prediction_start_time - training_start_time), (prediction_stop_time - prediction_start_time)))

min_sample_split = range(100,200,100)
for i in min_sample_split:

	# create estimator
	clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=i)
	
	# graph learning curve
	graph_learning_curve(clf, '{0} decision tree with pruning learning curve\n min_sample_split = {1}'.format(title, i), X, y )
	
	# start training clock
	training_start_time = time.perf_counter()

	clf.fit(X_train, y_train)

	# start prediction clock
	prediction_start_time = time.perf_counter()

	# Predict labels on the test set.
	y_pred =  clf.predict(X_test)
	prediction_stop_time = time.perf_counter()

	# print results
	print('\n\nPruned Decision Tree: min_sample_split = ', i)
	print('total training time was %.6f' % (prediction_start_time - training_start_time))
	print('total prediction time was %.6f' % (prediction_stop_time - prediction_start_time))

	y_pred_train=clf.predict(X_train)
	print('Accuracy Score on train data: %.4f' % (accuracy_score(y_true=y_train, y_pred=y_pred_train)))
	print('Accuracy Score on test data: %.4f' % (accuracy_score(y_true=y_test, y_pred=y_pred)))
	print('F1 Score Score on train data: %.4f' % (f1_score(y_true=y_train, y_pred=y_pred_train)))
	print('F1 Score Score on test data: %.4f' % (f1_score(y_true=y_test, y_pred=y_pred)))
	
	f.write("%.4f,%.4f,%.4f,%.4f,%.6f,%.6f" % ((accuracy_score(y_true=y_train, y_pred=y_pred_train)), (accuracy_score(y_true=y_test, y_pred=y_pred)), (f1_score(y_true=y_train, y_pred=y_pred_train)), (f1_score(y_true=y_test, y_pred=y_pred)), (prediction_start_time - training_start_time), (prediction_stop_time - prediction_start_time)))

	plt.show()

