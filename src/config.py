# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import numpy as np
from scipy.stats import randint as sp_randint, expon, truncnorm


#: str: Relative path to the datasets.
dataset = 'data/polypairs_dataset.csv'

#: int: Seed used by each of the random number generators.
seed_no = 42

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

    n_jobs = 4  #: int: Number of parallel jobs to be initiated. -1 means to utilize all available processors.

    #: bool: Whether to build additional features or not, i.e., convex hull of polygons and dist of centroids.
    extra_features = True

    # accepted values: randomized, grid, hyperband - not yet implemented!!!
    hyperparams_search_method = 'randomized'
    """str: Search Method to use for finding best hyperparameters. (*randomized* | *grid*).
    
    See Also
    --------
    :func:`~src.param_tuning.ParamTuning.fineTuneClassifiers`. Details on available inputs.       
    """

    #: int: Number of iterations that RandomizedSearchCV should execute. It applies only when :class:`hyperparams_
    #: search_method` equals to 'randomized'.
    max_iter = 500

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
            'gamma': 0.2456161956918959, 'max_iter': 3000, 'C': 199.0212721894755, 'kernel': 'rbf',
            'class_weight': None, 'tol': 0.0001, 'degree': 2,
            'random_state': seed_no
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
            'max_features': 7, 'max_depth': 50, 'min_samples_split': 0.5138831080474099,
            'min_samples_leaf': 0.12655804022659994, 'class_weight': {1: 1, 4: 4},
            'random_state': seed_no,
        },
        'RandomForest': {
            # with scaler
            # basic
            # 'max_depth': 62, 'min_samples_split': 2, 'n_estimators': 553, 'max_features': 'sqrt', 'bootstrap': False,
            # 'criterion': 'entropy', 'min_samples_leaf': 2, 'class_weight': {1: 1, 4: 7},
            # extra
            'max_depth': 77, 'min_samples_split': 8, 'min_samples_leaf': 2, 'bootstrap': False, 'criterion': 'entropy',
            'n_estimators': 459, 'max_features': 'log2', 'class_weight': {1: 1, 4: 7},
            # 'max_depth': 62, 'criterion': 'entropy', 'bootstrap': False, 'min_samples_split': 5,
            # 'class_weight': {1: 1, 4: 3}, 'n_estimators': 807, 'min_samples_leaf': 3, 'max_features': 'log2',
            'random_state': seed_no, 'n_jobs': n_jobs,  # 'oob_score': True,
        },
        'ExtraTrees': {
            # with scaler
            # basic
            # 'max_depth': 71, 'bootstrap': False, 'criterion': 'gini', 'class_weight': {1: 1, 4: 1},
            # 'min_samples_leaf': 1, 'max_features': 'sqrt', 'min_samples_split': 7, 'n_estimators': 776,
            # extra
            'class_weight': {1: 1, 4: 4}, 'max_depth': 98, 'criterion': 'gini', 'bootstrap': False,
            'min_samples_leaf': 1, 'max_features': 'sqrt', 'min_samples_split': 4, 'n_estimators': 887,
            'random_state': seed_no, 'n_jobs': n_jobs
        },
        'XGBoost': {
            # with scaler
            # basic
            # 'n_estimators': 2549, 'min_child_weight': 1, 'max_depth': 62, 'scale_pos_weight': 1,
            # 'colsample_bytree': 0.598605740971479, 'gamma': 1, 'eta': 0.17994840726392214,
            # 'subsample': 0.7250606565532803,
            # extra
            'n_estimators': 1638, 'colsample_bytree': 0.8721685761725149, 'min_child_weight': 2, 'gamma': 1,
            'max_depth': 79, 'eta': 0.03649597209843184, 'subsample': 0.957596136356163, 'scale_pos_weight': 1,
            'seed': seed_no, 'nthread': n_jobs
        },
        'MLP': {
            # with scaler
            # basic
            # 'activation': 'logistic', 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 0.0001,
            # 'learning_rate_init': 0.16533315728128767,
            # extra
            'tol': 0.0001, 'learning_rate_init': 0.06794912926673598, 'max_iter': 1000, 'activation': 'logistic',
            'solver': 'lbfgs',
            'random_state': seed_no,
        },
    }

    # These parameters constitute the search space for GridSearchCV in our experiments.
    SVM_hyperparameters = [
        {
            'kernel': ['rbf', 'sigmoid'],
            'gamma': [1e-2, 0.1, 1, 10, 100, 'scale'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'tol': [1e-3, 1e-4],
            'probability': [True, False],
            'max_iter': [3000],
            'class_weight': [None, 'balanced', {1: 1, 4: 2}, {1: 1, 4: 5}],
        },
        {
            'kernel': ['poly'],
            'gamma': [1e-2, 0.1, 1, 10, 100, 'scale'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'degree': [1, 2, 3],  # degree=1 is the same as using a 'linear' kernel
            'tol': [1e-3, 1e-4],
            'probability': [True, False],
            'max_iter': [3000],
            'class_weight': [None, 'balanced', {1: 1, 4: 2}, {1: 1, 4: 5}],
        },
        # {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [3000]}
    ]
    DecisionTree_hyperparameters = {
        'max_depth': [i for i in range(5, 30, 5)] + [None],
        'min_samples_split': [2, 4, 6, 10, 15, 25],
        'min_samples_leaf': [1, 2, 4, 10],
        # 'min_samples_split': list(np.linspace(0.1, 1, 10)),
        # 'min_samples_leaf': list(np.linspace(0.1, 0.5, 5)),
        'max_features': list(np.linspace(0.1, 1.0, 10)) + ['auto', 'log2'],
        'splitter': ('best', 'random'),
        'class_weight': [None, 'balanced', {1: 1, 4: 2}, {1: 1, 4: 5}],
    }
    RandomForest_hyperparameters = {
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 50, 60, 100, None],
        'criterion': ['gini', 'entropy'],
        'max_features': ['log2', 'sqrt'],  # auto is equal to sqrt
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        "n_estimators": [250, 500, 1000],
        'class_weight': [None, 'balanced', {1: 1, 4: 2}, {1: 1, 4: 5}],
    }
    XGBoost_hyperparameters = {
        "n_estimators": [500, 1000, 3000],
        # 'eta': list(np.linspace(0.01, 0.2, 10)),  # 'learning_rate'
        ## avoid overfitting
        # Control the model complexity
        'max_depth': [3, 5, 10, 30, 50, 70, 100],
        'gamma': [0, 1, 5],
        # 'alpha': [1, 10],
        # Add randomness to make training robust to noise
        'subsample': [0.8, 0.9, 1],
        'colsample_bytree': list(np.linspace(0.3, 1, 3)),
        # for class imbalances, setting the parameters
        # *) 'max_delta_step',
        # *) 'min_child_weight' and
        # *) 'scale_pos_weight'
        # could help
        'scale_pos_weight': [1, 2, 5, 10],
        'min_child_weight': [1, 5, 10],
    }
    MLP_hyperparameters = {
        'learning_rate_init': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'max_iter': [1000],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'tol': [1e-3, 1e-4],
    }

    # These parameters constitute the search space for RandomizedSearchCV in our experiments.
    SVM_hyperparameters_dist = {
        'C': expon(scale=100),
        'gamma': expon(scale=.1),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'class_weight': ['balanced', None] + [{1: 1, 4: w} for w in range(1, 10)],
        'degree': [1, 2, 3],
        'tol': [1e-3, 1e-4],
        'max_iter': [3000]
    }
    DecisionTree_hyperparameters_dist = {
        'max_depth': sp_randint(10, 100),
        'min_samples_split': truncnorm(0.1, 1),
        'min_samples_leaf': truncnorm(0.1, 0.5),
        'max_features': sp_randint(1, 11),
        'class_weight': [None, 'balanced'] + [{1: 1, 4: w} for w in range(1, 10)],
    }
    RandomForest_hyperparameters_dist = {
        'bootstrap': [True, False],
        'max_depth': sp_randint(3, 100),
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],  # sp_randint(1, 11)
        'min_samples_leaf': sp_randint(1, 50),
        'min_samples_split': sp_randint(2, 11),
        "n_estimators": sp_randint(250, 1000),
        'class_weight': ['balanced', None] + [{1: 1, 4: w} for w in range(1, 10)],
    }
    XGBoost_hyperparameters_dist = {
        "n_estimators": sp_randint(500, 4000),
        'eta': expon(loc=0.01, scale=0.1),  # 'learning_rate'
        # hyperparameters to avoid overfitting
        'max_depth': sp_randint(3, 100),
        'gamma': sp_randint(1, 5),
        'subsample': truncnorm(0.7, 1),
        'colsample_bytree': truncnorm(0, 1),
        'min_child_weight': sp_randint(1, 10),
        'scale_pos_weight': sp_randint(1, 30),
    }
    MLP_hyperparameters_dist = {
        'learning_rate_init': expon(loc=0.0001, scale=0.1),
        'max_iter': [1000],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'tol': [1e-3, 1e-4],
    }
