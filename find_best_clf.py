#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
import itertools
import random

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import datetime

import config

from  polygon_feature_extraction import *

np.random.seed(1234)

def get_scores(X_test, y_test, clf):
	
	y_pred = clf.predict(X_test)	
	
	#print(X_test.shape)
	probs = clf.predict_proba(X_test)
	
	return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro')


def fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test):
	
	#scores = ['precision', 'recall']
	scores = ['accuracy']#, 'f1_macro', 'f1_micro']
	
	if clf_name == "SVM":

		tuned_parameters = config.initialConfig.SVM_hyperparameters
		clf = SVC(probability=True)
		
	elif clf_name == "Nearest Neighbors":
		tuned_parameters = config.initialConfig.kNN_hyperparameters
		clf = KNeighborsClassifier()
		
	elif clf_name == "Decision Tree":

		tuned_parameters = config.initialConfig.DecisionTree_hyperparameters
		clf = DecisionTreeClassifier()
		
	elif clf_name == "Random Forest":

		tuned_parameters = config.initialConfig.RandomForest_hyperparameters
		clf = RandomForestClassifier()
	
	"""
	elif clf_name == "AdaBoost":
		tuned_parameters = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }
		clf = AdaBoostClassifier()
	
	elif clf_name == "MLP":
		tuned_parameters = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
		clf = MLPClassifier()
		
	elif clf_name == "Gaussian Process":
		
		clf = GaussianProcessClassifier()
	
	elif clf_name == "QDA":
		tuned_parameters = 
		clf = QuadraticDiscriminantAnalysis()
	"""
	
	print(clf_name)
		
	for score in scores:

		clf = GridSearchCV(clf, tuned_parameters, cv=4,
						   scoring='%s' % score, verbose=0)
		clf.fit(X_train, y_train)
		
	return clf
		
def tuned_parameters_5_fold(args):
	
	# Get X, Y data
	print(args['polygon_file_name'])
	X, y = get_X_Y_data(args['polygon_file_name'])
	#np.savetxt("X", X, delimiter=",")
	#np.savetxt("y", y, delimiter=",")
	
	#X = np.loadtxt('X.csv', delimiter=",")
	#y = np.loadtxt('y.csv', delimiter=",")
	print(X.shape, y.shape)
	skf = StratifiedKFold(n_splits = config.initialConfig.k_fold_parameter)
	
	count = 1
	
	clf_names_not_tuned = ["Naive Bayes", "MLP", "Gaussian Process", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
	for item in clf_scores_dict:
		clf_scores_dict[item] = [[], [], []]
		
	report_data = []
	hyperparams_data = []
	
	#print(poi_ids)
	
	# split data into train, test
	for train_index, test_index in skf.split(X, y):

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
						
		for clf_name in clf_names:
			row = {}
			#print(clf_name)
			
			if clf_name in clf_names_not_tuned:
				if clf_name == "Naive Bayes":
					clf = GaussianNB()
					clf.fit(X_train, y_train)
				elif clf_name == "MLP":
					clf = MLPClassifier()
					clf.fit(X_train, y_train)
				elif clf_name == "Gaussian Process":
					clf = GaussianProcessClassifier()
					clf.fit(X_train, y_train)
				#elif clf_name == "QDA":
				#	clf = QuadraticDiscriminantAnalysis()
				#	clf.fit(X_train, y_train)
				else:
					clf = AdaBoostClassifier()
					clf.fit(X_train, y_train)
			else:
				clf = fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test)
			
			accuracy, f1_score_micro, f1_score_macro = get_scores(X_test, y_test, clf)
			clf_scores_dict[clf_name][0].append(accuracy)
			clf_scores_dict[clf_name][1].append(f1_score_micro)
			clf_scores_dict[clf_name][2].append(f1_score_macro)
			row['Fold'] = count
			row['Classifier'] = clf_name
			row['Accuracy'] = accuracy
			row['F1-Score-Micro'] = f1_score_micro
			row['F1-Score-Macro'] = f1_score_macro
			
			if clf_name not in clf_names_not_tuned:
				hyperparams_row = {}
				hyperparams_row['Fold'] = count
				hyperparams_row['Classifier'] = clf_name
				hyperparams_row['Best Hyperparameters'] = clf.best_params_
				hyperparams_data.append(hyperparams_row)
				
			report_data.append(row)
			
		count += 1
		
	for clf_name in clf_names:	
		row = {}
		row['Fold'] = 'average'
		row['Classifier'] = clf_name
		row['Accuracy'] = sum(map(float,clf_scores_dict[clf_name][0])) / 5.0 
		row['F1-Score-Micro'] = sum(map(float,clf_scores_dict[clf_name][1])) / 5.0
		row['F1-Score-Macro'] = sum(map(float,clf_scores_dict[clf_name][2])) / 5.0
		report_data.append(row)
		
	df = pd.DataFrame.from_dict(report_data)
	if args['results_file_name'] is not None:
		filename = args['results_file_name'] + '_' + str(datetime.datetime.now()) + '.csv'
	else:
		filename = 'classification_report_' + str(datetime.datetime.now()) + '.csv'
	df.to_csv(filename, index = False)
	
	best_clf_row = {}
	best_clf_row['best_clf_score'] = 0.0
	best_clf_row['best_clf_name'] = ''
	for index, row in df.iterrows():
		if row['Fold'] == 'average':
			if row['Accuracy'] > best_clf_row['best_clf_score'] :
				best_clf_row['best_clf_score']  = row['Accuracy']
				best_clf_row['best_clf_name'] = row['Classifier']
	df2 = pd.DataFrame.from_dict([best_clf_row])
	filename = 'best_clf_' + '_' + str(datetime.datetime.now()) + '.csv'
	df2.to_csv(filename, index = False)
	
	df3 = pd.DataFrame.from_dict(hyperparams_data)
	if args['hyperparameter_file_name'] is not None:
		filename = args['hyperparameter_file_name'] + '_' + str(datetime.datetime.now()) + '.csv'
	else:
		filename = 'hyperparameters_per_fold_' + str(datetime.datetime.now()) + '.csv'
	df3.to_csv(filename, index = False)
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-polygon_file_name", "--polygon_file_name", required=False,
		help="name of table containing pois information")
	ap.add_argument("-results_file_name", "--results_file_name", required=False,
		help="desired name of output file")
	ap.add_argument("-hyperparameter_file_name", "--hyperparameter_file_name", required=False,
		help="desired name of output file")

	args = vars(ap.parse_args())

	tuned_parameters_5_fold(args)
	
	
if __name__ == "__main__":
   main()
