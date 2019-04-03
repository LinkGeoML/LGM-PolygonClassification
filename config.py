#!/usr/bin/python
import numpy as np

class initialConfig:
	
	# The following parameters correspond to the machine learning
	# part of the framework.
	
	# This parameter refers to the number of outer folds that
	# are being used in order for the k-fold cross-validation
	# to take place.
	k_fold_parameter = 5
	
	# This parameter contains a list of the various classifiers
	# the results of which will be compared in the experiments.
	classifiers = ['Nearest Neighbors', 'SVM', 'Decision Tree', 'Random Forest', 'AdaBoost', 
	'Naive Bayes', 'MLP', 'Gaussian Process']
	#classifiers = ['AdaBoost']
	
	# These are the parameters that constitute the search space
	# in our experiments.
	kNN_hyperparameters = {"n_neighbors": [1, 3, 5, 10, 20]}
	SVM_hyperparameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
						 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
						 {'kernel': ['poly'],
                             'degree': [1, 2, 3, 4],
                             'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
						{'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
	DecisionTree_hyperparameters = {'max_depth': [i for i in range(1,33)], 
		'min_samples_split': list(np.linspace(0.1,1,10)),
		'min_samples_leaf': list(np.linspace(0.1,0.5,5)),
                  'max_features': [i for i in range(1, 10)]}
	RandomForest_hyperparameters = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'criterion': ['gini', 'entropy'],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 "n_estimators": [250, 500, 1000]}
 
	# The following parameters refer to the names of the csv columns
	# that correspond to the following: poi_id (the unique id of each 
	# poi), name (the name of each poi), class_codes (list of the class 
	# level names that we want to be considered )
	# x (the longitude of the poi), y (the latitude of the poi)
	
	#poi_id = "poi_id"
	poi_id = "id"
	name = "name"
	class_codes = ["class_name"]
	#class_codes = ["theme", "class_name", "subclass_n"]
	x = "y"
	y = "x"
	
	# The following parameter corresponds to the original SRID of the
	# poi dataset that is being fed to the experiments.
	
	#original_SRID = 'epsg:2100'
	original_SRID = 'epsg:3857'
