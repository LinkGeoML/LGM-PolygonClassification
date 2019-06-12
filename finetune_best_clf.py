#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
import nltk
import itertools
import random

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

from  polygon_feature_extraction import *

import csv
import datetime
import config
import glob
import os

np.random.seed(1234)

def get_scores(X_test, y_test, clf):
	
	"""
	This function is responsible for measuring the prediction scores
	during the test phase
	
	Arguments
	---------
	X_test: :obj:`numpy array`
		the test set features
	y_test: :obj:`numpy array`
		the test set class labels
	most_common_classes: :obj:`list`
		the 10 most common (most populated) classes
	clf: :obj:`scikit-learn classifier object`
		the classifier object that is used for the predictions
	
	Returns
	-------
	accuracy_score(y_test, y_pred): :obj:`float`
		the accuracy score as computed by scikit-learn
	f1_score(y_test, y_pred, average='weighted'): :obj:`float`
		the weighted f1-score as computed by scikit-learn
	f1_score(y_test, y_pred, average='macro'): :obj:`float`
		the macro f1-score as computed by scikit-learn
	"""
	
	y_pred = clf.predict(X_test)	
	
	#print(X_test.shape)
	probs = clf.predict_proba(X_test)
	
	return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro')

def fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test):
	
	"""
	This function is responsible for fitting a classifier
	to a training set and returning the classifier object
	for later use.
	
	Arguments
	---------
	X_train: :obj:`numpy array`
		array containing the features of the train set
	y_train: :obj:`numpy array`
		array containing the labels of the train set
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes 
	
	Returns
	-------
	clf: :obj:`scikit-learn classifier object`
		the trained classifier object
	"""
	
	scores = ['accuracy']
	
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
		
	elif clf_name == "Extra Trees":

		tuned_parameters = config.initialConfig.RandomForest_hyperparameters
		clf = ExtraTreesClassifier()
	
	elif clf_name == "MLP":
		
		tuned_parameters = config.initialConfig.MLP_hyperparameters
		clf = MLPClassifier()
	
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
	
	"""
	This function trains a collection of classifiers using
	a nested k-fold cross-validation approach and outputs
	the relevant results so that later comparisons can be made
	
	Arguments
	---------
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes 
	
	Returns
	-------
	None
	"""
	
	# Shuffle ids
	
	X, y = get_X_Y_data(args['polygon_file_name'])
			
	clf_names_not_tuned = ["Naive Bayes", "Gaussian Process", "QDA", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
	for item in clf_scores_dict:
		clf_scores_dict[item] = [[], [], []]
	report_data = []
	hyperparams_data = []
			
	# get train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
			
	# read clf name from csv
	row = {}
	
	if args['best_clf'] in clf_names_not_tuned:
		if clf_name == "Naive Bayes":
			clf = GaussianNB()
			clf.fit(X_train, y_train)
		#elif clf_name == "MLP":
		#	clf = MLPClassifier()
		#	clf.fit(X_train, y_train)
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
		clf = fine_tune_parameters_given_clf(args['best_clf'], X_train, y_train, X_test, y_test)
	
	hyperparams_data = clf.best_params_
	df2 = pd.DataFrame.from_dict([hyperparams_data])
	#print(hyperparams_data)
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'best_hyperparameters.csv'
			df2.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'best_hyperparameters.csv'
		df2.to_csv(filepath, index = False)
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-polygon_file_name", "--polygon_file_name", required=False,
		help="name of table containing pois information")
	#ap.add_argument("-results_file_name", "--results_file_name", required=False,
	#	help="desired name of best hyperparameter file")
	ap.add_argument("-best_hyperparameter_file_name", "--best_hyperparameter_file_name", required=False,
		help="desired name of best hyperparameter file")
	ap.add_argument("-best_clf_file_name", "--best_clf_file_name", required=False,
		help="name of file containing the best classifier found in step 1")

	args = vars(ap.parse_args())

	if config.initialConfig.experiment_folder is not None:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		exists = os.path.isdir(experiment_folder_path)
		if exists:
			filepath = experiment_folder_path + '/' + 'best_clf.csv'
			exists2 = os.path.isfile(filepath)
			if exists2:
				with open(filepath, 'r') as csv_file:
					reader = csv.reader(csv_file)
					count = 0
					for row in reader:
						if count == 1:
							args['best_clf'] = row[0]
						count += 1
			else:
				print("ERROR! No best_clf file found inside the folder")
				return
		else:
			print("ERROR! No experiment folder with the given name found")
			return
	else:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'best_clf.csv'
			exists = os.path.isfile(filepath)
			if exists:
				with open(filepath, 'r') as csv_file:
					reader = csv.reader(csv_file)
					count = 0
					for row in reader:
						if count == 1:
							args['best_clf'] = row[0]
						count += 1
			else:
				print("ERROR! No best_clf file found inside the folder!")	
				return
			
	tuned_parameters_5_fold(args)
	
if __name__ == "__main__":
   main()
