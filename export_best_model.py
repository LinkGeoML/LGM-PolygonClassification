#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from pois_feature_extraction import *
from textual_feature_extraction import *
from feml import *
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

import glob
import os

np.random.seed(1234)
		
def tuned_parameters_5_fold(args):
	
	from sklearn.externals import joblib
	
	X_train, y_train = get_X_Y_data()
			
	clf_names_not_tuned = ["Naive Bayes", "MLP", "Gaussian Process", "QDA", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
	
	# read clf name from csv
	row = {}
	
	if args['best_clf'] in clf_names_not_tuned:
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
		# read hyperparameters and fit the model
		clf = train_clf_given_hyperparams(X_train, y_train, args)
		
	# store the model as a pickle file
	
	if args['trained_model_file_name'] is not None:
		filename = args['trained_model_file_name'] + '_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.pkl'
		joblib.dump(clf, filename, compress = 9)
	else:
		filename = 'trained_model_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.pkl'
		joblib.dump(clf, filename, compress = 9)

def train_clf_given_hyperparams(X_train, y_train, args):
	tuned_parameters = args['best_hyperparams']
	for parameter in tuned_parameters:
		#print(parameter)
		if tuned_parameters[parameter].isdigit():
			#print("edw")
			tuned_parameters[parameter] = float(tuned_parameters[parameter])
	#tuned_parameters['degree'] = int(tuned_parameters['degree'])
	print(tuned_parameters)
	
	if args['best_clf'] == "SVM":
		clf = SVC(probability=True, **tuned_parameters)
		
	elif args['best_clf'] == "Nearest Neighbors":
		clf = KNeighborsClassifier(**tuned_parameters)
		
	elif args['best_clf'] == "Decision Tree":
		clf = DecisionTreeClassifier(**tuned_parameters)
		
	elif args['best_clf'] == "Random Forest":
		clf = RandomForestClassifier(**tuned_parameters)
	
	print(clf)

	clf.fit(X_train, y_train)
		
	return clf
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-geocoding_file_name", "--geocoding_file_name", required=False,
		help="name of table containing pois information")
	#ap.add_argument("-results_file_name", "--results_file_name", required=False,
	#	help="desired name of best hyperparameter file")
	ap.add_argument("-best_hyperparameter_file_name", "--best_hyperparameter_file_name", required=False,
		help="desired name of best hyperparameter file")
	ap.add_argument("-best_clf_file_name", "--best_clf_file_name", required=False,
		help="desired name of best hyperparameter file")
	ap.add_argument("-trained_model_file_name", "--trained_model_file_name", required=False,
		help="name of file containing the trained model")

	args = vars(ap.parse_args())
	
	if args['best_clf_file_name'] is not None:
		#with open(args['best_clf_file_name']) as f:
		#	args['best_clf'] = f.readline()
		input_file = csv.DictReader(open(args['best_clf_file_name']))
		with open(input_file, 'r') as csv_file:
			reader = csv.reader(csv_file)
			count = 0
			for row in reader:
				if count == 1:
					args['best_clf'] = row[0]
				count += 1
	else:
		list_of_files = glob.glob('best_clf_*')
		input_file = max(list_of_files, key=os.path.getctime)
		with open(input_file, 'r') as csv_file:
			reader = csv.reader(csv_file)
			count = 0
			for row in reader:
				if count == 1:
					args['best_clf'] = row[0]
				count += 1
			
	#print(args['best_clf'])
	
	if args['best_hyperparameter_file_name'] is not None:
		input_file = csv.DictReader(open(args['best_hyperparameter_file_name']))
		for row in input_file:
			hyperparams_dict = row
	else:
		list_of_files = glob.glob('best_hyperparameters_*')
		latest_file = max(list_of_files, key=os.path.getctime)
		input_file = csv.DictReader(open(latest_file))
		for row in input_file:
			hyperparams_dict = row
	
	args['best_hyperparams'] = hyperparams_dict
	
	tuned_parameters_5_fold(args)
	
	
if __name__ == "__main__":
   main()
