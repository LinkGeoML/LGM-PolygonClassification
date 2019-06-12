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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import datetime

import config
import glob
import os

from  polygon_feature_extraction import *

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
	clf_name: :obj:`str`
		name of the classifier
	X_train: :obj:`numpy array`
		array containing the features of the train set
	y_train: :obj:`numpy array`
		array containing the labels of the train set
	X_test: :obj:`numpy array`
		array containing the features of the test set
	y_test: :obj:`numpy array`
		array containing the labels of the test set
	
	Returns
	-------
	clf: :obj:`scikit-learn classifier object`
		the trained classifier object
	"""
	
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
	
	# Get X, Y data
	print(args['polygon_file_name'])
	X, y = get_X_Y_data(args['polygon_file_name'])
	#np.savetxt("X", X, delimiter=",")
	#np.savetxt("y", y, delimiter=",")
	
	print(X.shape)
	
	feature_dict = {}
	feature_dict['area'] = X[:, 0:1]
	feature_dict['insersection_area'] = X[:, 1:2]
	feature_dict['perimeter'] = X[:, 2:3]
	feature_dict['vertex_count'] = X[:, 3:4]
	feature_dict['mean_edge_length'] = X[:, 4:5]
	feature_dict['variance_edge_length'] = X[:, 5:6]
	
	scaler_dict = dict((el, None) for el in config.initialConfig.included_features)
	sentinel = 0
	for key in feature_dict:
		if feature_dict[key] is not None:
			if sentinel == 0:
				X = np.asarray(feature_dict[key])
				print("Pre Normalization - Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(X), np.std(X), np.amax(X), np.amin(X), X.shape))
				if key in config.initialConfig.included_features:
					X, scaler_dict[key] = standardize_data_train(X)
				print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(X), np.std(X), np.amax(X), np.amin(X), X.shape))
				sentinel = 1
			else:
				temp_array = np.asarray(feature_dict[key])
				print("Pre Normalization - Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(temp_array), np.std(temp_array), np.amax(temp_array), np.amin(temp_array), temp_array.shape))
				if key in config.initialConfig.included_features:
					temp_array, scaler_dict[key] = standardize_data_train(temp_array)
				print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(temp_array), np.std(temp_array), np.amax(temp_array), np.amin(temp_array), temp_array.shape))
				X = np.concatenate((X, temp_array), axis = 1)
	
	#return
	"""
	X = []
	sentinel = 0
	for key in features:
		temp_array = features[key]
		print("Pre Normalization - Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(temp_array), np.std(temp_array), np.amax(temp_array), np.amin(temp_array), temp_array.shape))
		temp_array, _ = standardize_data_train(temp_array)
		#filepath = config.initialConfig.root_path + key + '.csv'
		#np.savetxt(filepath, total_features, delimiter=",")
		print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(temp_array), np.std(temp_array), np.amax(temp_array), np.amin(temp_array), temp_array.shape))
	"""
		
	#return
	
	#X = np.loadtxt('X.csv', delimiter=",")
	#y = np.loadtxt('y.csv', delimiter=",")
	print(X.shape, y.shape)
	skf = StratifiedKFold(n_splits = config.initialConfig.k_fold_parameter)
	
	count = 1
	
	clf_names_not_tuned = ["Naive Bayes", "Gaussian Process", "AdaBoost"]
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
	
	if config.initialConfig.experiment_folder == None:
		folderpath = config.initialConfig.root_path + 'experiment_folder_' + str(datetime.datetime.now())
		folderpath = folderpath.replace(':', '-')
		os.makedirs(folderpath)
		filepath = folderpath + '/' + 'classification_report.csv'
		df.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'classification_report.csv'
		df.to_csv(filepath, index = False)
	
	best_clf_row = {}
	best_clf_row['best_clf_score'] = 0.0
	best_clf_row['best_clf_name'] = ''
	for index, row in df.iterrows():
		if row['Fold'] == 'average':
			if row['Accuracy'] > best_clf_row['best_clf_score'] :
				best_clf_row['best_clf_score']  = row['Accuracy']
				best_clf_row['best_clf_name'] = row['Classifier']
	df2 = pd.DataFrame.from_dict([best_clf_row])

	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'best_clf.csv'
			df2.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'best_clf.csv'
		df2.to_csv(filepath, index = False)
	df2.to_csv(filepath, index = False)
	
	df3 = pd.DataFrame.from_dict(hyperparams_data)
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'hyperparameters_per_fold.csv'
			df3.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'hyperparameters_per_fold.csv'
		df3.to_csv(filepath, index = False)
	
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
