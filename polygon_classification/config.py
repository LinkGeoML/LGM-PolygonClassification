# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import numpy as np
from scipy.stats import randint as sp_randint, expon, truncnorm


#: str: Relative path to the datasets.
dataset = 'data/polypairs_dataset.csv'

#: int: Seed used by each of the random number generators.
seed_no = 2020

#: float: Proportion of the dataset to include in the test split. Accepted values should be between 0.0 and 1.0.
test_split_thres = 0.2


class MLConf:
    """
    This class initializes parameters that correspond to the machine learning part of the framework.

    These variables define the parameter grid for GridSearchCV:

    :cvar SVM_hyperparameters: Defines the search space for SVM.
    :vartype SVM_hyperparameters: :obj:`list`
    :cvar MLP_hyperparameters: Defines the search space for MLP.
    :vartype MLP_hyperparameters: :obj:`dict`
    :cvar DecisionTree_hyperparameters: Defines the search space for Decision Trees.
    :vartype DecisionTree_hyperparameters: :obj:`dict`
    :cvar RandomForest_hyperparameters: Defines the search space for Random Forests and Extra-Trees.
    :vartype RandomForest_hyperparameters: :obj:`dict`
    :cvar XGBoost_hyperparameters: Defines the search space for XGBoost.
    :vartype XGBoost_hyperparameters: :obj:`dict`

    These variables define the parameter grid for RandomizedSearchCV where continuous distributions are used for
    continuous parameters (whenever this is feasible):

    :cvar SVM_hyperparameters_dist: Defines the search space for SVM.
    :vartype SVM_hyperparameters_dist: :obj:`dict`
    :cvar MLP_hyperparameters_dist: Defines the search space for MLP.
    :vartype MLP_hyperparameters_dist: :obj:`dict`
    :cvar DecisionTree_hyperparameters_dist: Defines the search space for Decision Trees.
    :vartype DecisionTree_hyperparameters_dist: :obj:`dict`
    :cvar RandomForest_hyperparameters_dist: Defines the search space for Random Forests and Extra-Trees.
    :vartype RandomForest_hyperparameters_dist: :obj:`dict`
    :cvar XGBoost_hyperparameters_dist: Defines the search space for XGBoost.
    :vartype XGBoost_hyperparameters_dist: :obj:`dict`
    """

    kfold_parameter = 5  #: int: The number of outer folds that splits the dataset for the k-fold cross-validation.

    n_jobs = -1  #: int: Number of parallel jobs to be initiated. -1 means to utilize all available processors.

    #: bool: Whether to build additional features or not, i.e., convex hull of polygons and dist of centroids.
    extra_features = True

    # accepted values: randomized, grid, hyperband - not yet implemented!!!
    hyperparams_search_method = 'grid'
    """str: Search Method to use for finding best hyperparameters. (*randomized* | *grid*).
    
    See Also
    --------
    :func:`~src.param_tuning.ParamTuning.fineTuneClassifiers`. Details on available inputs.       
    """

    #: int: Number of iterations that RandomizedSearchCV should execute. It applies only when :class:`hyperparams_
    #: search_method` equals to 'randomized'.
    max_iter = 300

    score = 'accuracy'
    """str: The metric to optimize on hyper-parameter tuning. Possible valid values presented on `Scikit predefined values`_. 
    
    .. _Scikit predefined values:
        https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
    """

    classifiers = [
        # 'SVM',
        # 'DecisionTree',
        'RandomForest',
        # 'ExtraTrees',
        # 'XGBoost',
        # 'MLP'
    ]
    """list of str: Define the classifiers to apply on code execution. Accepted values are: 
        
    - SVM 
    - DecisionTree
    - RandomForest
    - ExtraTrees
    - XGBoost
    - MLP.
    """

    clf_custom_params = {
        'SVM': {
            # with scaler
            # basic/extra
            'C': 200, 'class_weight': 'balanced', 'gamma': 10, 'max_iter': 10000,
            # 'kernel': 'rbf', 'tol': 0.001
        },
        'DecisionTree': {
            # 'min_samples_leaf': 0.10218472045491575, 'min_samples_split': 0.46848801022523695, 'max_features': 10,
            # 'class_weight': {1: 1, 4: 9}, 'max_depth': 70,
            # with scaler
            # 'max_features': 10, 'min_samples_leaf': 0.13084191454406335, 'class_weight': {1: 1, 4: 2}, 'max_depth': 79,
            # 'min_samples_split': 0.7040970996893269,
            # basic
            # 'min_samples_leaf': 0.13896623393837215, 'class_weight': {1: 1, 4: 7}, 'max_depth': 42, 'max_features': 5,
            # 'min_samples_split': 0.21705549723971926,
            # extra
            'class_weight': {1: 1, 4: 2}, 'max_depth': 80, 'max_features': 10, 'min_samples_leaf': 2,
            'min_samples_split': 10,
            # 'splitter': 'best',
        },
        'RandomForest': {
            # with scaler
            # basic
            # 'max_depth': 62, 'min_samples_split': 2, 'n_estimators': 553, 'max_features': 'sqrt', 'bootstrap': False,
            # 'criterion': 'entropy', 'min_samples_leaf': 2, 'class_weight': {1: 1, 4: 7},
            # extra
            'class_weight':  {1: 2, 4: 1}, 'criterion': 'entropy', 'max_depth': 100, 'n_estimators': 1000,
            # 'min_samples_split': 2,
            # 'random_state': seed_no, 'n_jobs': n_jobs,  # 'oob_score': True,
        },
        'ExtraTrees': {
            # with scaler
            # basic
            # 'max_depth': 71, 'bootstrap': False, 'criterion': 'gini', 'class_weight': {1: 1, 4: 1},
            # 'min_samples_leaf': 1, 'max_features': 'sqrt', 'min_samples_split': 7, 'n_estimators': 776,
            # extra
            'class_weight': {1: 2, 4: 1}, 'max_depth': 100,
            # 'random_state': seed_no, 'n_jobs': n_jobs
        },
        'XGBoost': {
            # with scaler
            # basic
            # 'n_estimators': 2549, 'min_child_weight': 1, 'max_depth': 62, 'scale_pos_weight': 1,
            # 'colsample_bytree': 0.598605740971479, 'gamma': 1, 'eta': 0.17994840726392214,
            # 'subsample': 0.7250606565532803,
            # extra
            'max_depth': 72, 'n_estimators': 21, 'scale_pos_weight': 3,
            # 'random_state': seed_no, 'nthread': n_jobs, 'objective': "binary:logistic",
        },
        'MLP': {
            # with scaler
            # basic
            # 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 0.0001,
            # 'learning_rate_init': 0.16533315728128767,
            # extra
            # 'activation': 'relu', 'hidden_layer_sizes': (100,),
            'learning_rate_init': 0.05, 'max_iter': 5000, 'solver': 'lbfgs', 'tol': 0.003,
        }
    }

    # These parameters constitute the search space for GridSearchCV in our experiments.
    SVM_hyperparameters = [
        {
            'kernel': ['rbf', 'sigmoid'],
            'gamma': [1e-2, 0.1, 1, 5, 10, 30, 'scale'],
            'C': [0.01, 0.1, 1, 10, 100, 200, 300],
            'tol': [1e-3, 1e-2],
            # 'probability': [True, False],
            'max_iter': [5000],
            'class_weight': [None, 'balanced', {1: 2, 4: 1}, {1: 3, 4: 1}],
        },
        {
            'kernel': ['poly'],
            'gamma': ['auto', 'scale', 1, 10, 30],
            'C': [0.01, 0.1, 1, 10, 100, 200, 300],
            'degree': [1, 2, 3],  # degree=1 is the same as using a 'linear' kernel
            'tol': [1e-3, 1e-2],
            # 'probability': [True, False],
            'max_iter': [5000],
            'class_weight': [None, 'balanced', {1: 2, 4: 1}, {1: 3, 4: 1}],
        },
        # {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [3000]}
    ]
    DecisionTree_hyperparameters = {
        'max_depth': [2, 3, 5, 10, 30, 50, 60, 80, 100],
        'min_samples_split': [2, 4, 6, 10, 15, 25, 50],
        'min_samples_leaf': [1, 2, 4, 10],
        # 'min_samples_split': list(np.linspace(0.1, 1, 10)),
        # 'min_samples_leaf': list(np.linspace(0.1, 0.5, 5)),
        'max_features': list(np.arange(2, 11, 2)) + ["sqrt", "log2"],
        'splitter': ['best', 'random'],
        'class_weight': [None, 'balanced', {1: 2, 4: 1}, {1: 3, 4: 1}],
    }
    RandomForest_hyperparameters = {
        # 'bootstrap': [True, False],
        'max_depth': [5, 10, 20, 50, 70, 100],
        'criterion': ['gini', 'entropy'],
        # 'max_features': ['log2', 'sqrt'],  # auto is equal to sqrt
        # 'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 3, 5],
        "n_estimators": [100, 120, 200, 250],
        'class_weight': [None, 'balanced', {1: 2, 4: 1}, {1: 3, 4: 1}],
    }
    XGBoost_hyperparameters = {
        "n_estimators": [20, 30, 100, 300],
        # 'eta': list(np.linspace(0.01, 0.2, 10)),  # 'learning_rate'
        ## avoid overfitting
        # Control the model complexity
        'max_depth': [10, 30, 50, 70, 100],
        'gamma': [0, 1, 5],
        # 'reg_alpha': [1, 10],
        # Add randomness to make training robust to noise
        'subsample': [0.8, 0.9, 1],
        # 'colsample_bytree': list(np.linspace(0.8, 1, 3)),
        # for class imbalances, setting the parameters
        # *) 'max_delta_step',
        # *) 'min_child_weight' and
        # *) 'scale_pos_weight'
        # could help
        'scale_pos_weight': [1, 2, 3],
        # 'min_child_weight': [1, 5, 10],
    }
    MLP_hyperparameters = {
        'hidden_layer_sizes': [(100,), (50, 50,)],
        'learning_rate_init': [0.0001, 0.005, 0.01, 0.05, 0.1],
        'max_iter': [3000],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'tol': [1e-3, 1e-4],
    }

    # These parameters constitute the search space for RandomizedSearchCV in our experiments.
    SVM_hyperparameters_dist = {
        'C': expon(scale=100),
        'gamma': expon(scale=.1),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'class_weight': ['balanced', None] + [{1: w, 4: 1} for w in range(1, 5)],
        'degree': [1, 2, 3],
        'tol': [1e-3, 1e-2],
        'max_iter': [10000]
    }
    DecisionTree_hyperparameters_dist = {
        'max_depth': sp_randint(10, 200),
        'min_samples_split': sp_randint(2, 200),
        'min_samples_leaf': sp_randint(1, 10),
        'max_features': sp_randint(1, 11),
        'class_weight': [None, 'balanced'] + [{1: w, 4: 1} for w in range(1, 5)],
    }
    RandomForest_hyperparameters_dist = {
        # 'bootstrap': [True, False],
        'max_depth': sp_randint(3, 200),
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],  # sp_randint(1, 11)
        'min_samples_leaf': sp_randint(1, 10),
        'min_samples_split': sp_randint(2, 21),
        "n_estimators": sp_randint(250, 1000),
        'class_weight': ['balanced', None] + [{1: w, 4: 1} for w in range(1, 5)],
    }
    XGBoost_hyperparameters_dist = {
        "n_estimators": sp_randint(20, 200),
        # 'eta': expon(loc=0.01, scale=0.1),  # 'learning_rate'
        # hyperparameters to avoid overfitting
        'max_depth': sp_randint(10, 200),
        'gamma': sp_randint(0, 5),
        'subsample': truncnorm(0.4, 0.7),
        # 'colsample_bytree': truncnorm(0.8, 1),
        # 'min_child_weight': sp_randint(1, 10),
        'scale_pos_weight': sp_randint(1, 5),
        "reg_alpha": truncnorm(0, 2),
        'reg_lambda': sp_randint(1, 20),
    }
    MLP_hyperparameters_dist = {
        'hidden_layer_sizes': [(100,), (50, 50,)],
        'learning_rate_init': expon(loc=0.0001, scale=0.1),
        'max_iter': [3000],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'tol': [1e-3, 1e-4],
    }
