'''
# neural_net.py
# 
# Runs neural net model 
#
# Neal Messer
# nmesser3@gatech.edu
# 
# Skeleton of code and basic information on neural nets from:
# https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
#
# Code modified for GA Tech OMSCS 7641 Machine Learning Class
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time, sys
from sklearn.neural_network import MLPClassifier
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
	print ('Choose dataset: income OR personality\nUsage: neural_net.py income OR neural_net.py personality')
	exit(1)


# read in training file
data = pd.read_csv(input_file)

# set up training data
y = data[output_col]       
X = data.drop(columns=[output_col])

# use the train_test_split to create train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 12, test_size = 1/3)

# scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X = scaler.transform(X)

# train neural net
mlp = MLPClassifier()

# graph learning curve
graph_learning_curve(mlp, '{0} neural net learning curve'.format(title), X, y )


# start training clock
training_start_time = time.perf_counter()
mlp.fit(X_train,y_train)

# start prediction clock
prediction_start_time = time.perf_counter()
	   
y_pred = mlp.predict(X_test)

prediction_stop_time = time.perf_counter()

print('Neural Net')
print('total training time was %.6f' % (prediction_start_time - training_start_time))
print('total prediction time was %.6f' % (prediction_stop_time - prediction_start_time))

y_pred_train=mlp.predict(X_train)
print('Accuracy Score on train data: %.4f' % (accuracy_score(y_true=y_train, y_pred=y_pred_train)))
print('Accuracy Score on test data: %.4f' % (accuracy_score(y_true=y_test, y_pred=y_pred)))
print('F1 Score Score on train data: %.4f' % (f1_score(y_true=y_train, y_pred=y_pred_train)))
print('F1 Score Score on test data: %.4f' % (f1_score(y_true=y_test, y_pred=y_pred))) 

out_file = "nn%d.csv" % time.time()
f = open(out_file, "a")
f.write("%.4f,%.4f,%.4f,%.4f,%.6f,%.6f\n" % ((accuracy_score(y_true=y_train, y_pred=y_pred_train)), (accuracy_score(y_true=y_test, y_pred=y_pred)), (f1_score(y_true=y_train, y_pred=y_pred_train)), (f1_score(y_true=y_test, y_pred=y_pred)), (prediction_start_time - training_start_time), (prediction_stop_time - prediction_start_time)))


# Cost plotting based on answer to StackOverflow question at
# https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-mlpclassifier
plt.figure()
plt.title(title + ' Neural Net Cost Curve' )
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.plot(mlp.loss_curve_)
plt.show()