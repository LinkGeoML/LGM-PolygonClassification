#!/usr/bin/python
import numpy as np

class initialConfig:
	
	feature_list = ['area', 'insersection_area_percentage', 'perimeter', 
	'vertex_count', 'mean_edge_length', 'variance_edge_length']
	
	included_features = ['area', 'insersection_area_percentage', 'perimeter', 
	'vertex_count', 'mean_edge_length', 'variance_edge_length']
	
	features_to_normalize = ['area', 'insersection_area_percentage', 'perimeter', 
	'vertex_count', 'mean_edge_length', 'variance_edge_length']
	
	root_path = "/home/nikos/Desktop/LGM-PolygonClassification/"
	#experiment_folder = "experiment_folder_2019-04-08 16-07-31.806855"
	experiment_folder = None
	
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
	#classifiers = ['Nearest Neighbors']
	
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
		'min_samples_split': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
		'min_samples_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
                  'max_features': [i for i in range(1, 6)]}
	RandomForest_hyperparameters = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'criterion': ['gini', 'entropy'],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 "n_estimators": [250, 500, 1000]}
	MLP_hyperparameters = {'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
							'max_iter': [100, 200, 500, 1000],
							'solver': ['sgd', 'adam']}
 
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
