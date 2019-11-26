# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import time
from sklearn.model_selection import train_test_split
import geopandas as gpd

from src import param_tuning, config
from src.features import Features
from src.helpers import getRelativePathtoWorking


class StrategyEvaluator:
    """
    This class implements the pipeline for various strategies.
    """
    def __init__(self):
        pass

    def hyperparamTuning(self):
        """A complete process of distinct steps in figuring out the best ML algorithm with best hyperparameters to
        polygon classification problem.
        """
        pt = param_tuning.ParamTuning()
        f = Features()

        tot_time = time.time(); start_time = time.time()
        Xtrain, Xtest, ytrain, ytest = self._load_and_split_data()
        print("Loaded train/test datasets in {} sec.".format(time.time() - start_time))

        fX = f.build(Xtrain)
        print("Build features from train data in {} sec.".format(time.time() - start_time))

        start_time = time.time()
        # 1st phase: find and fine tune the best classifier from a list of candidate ones
        best_clf, estimator = pt.fineTuneClassifiers(fX, ytrain)
        estimator = best_clf['estimator']
        print("Best hyperparams, {}, with score {}; {} sec.".format(
            best_clf['hyperparams'], best_clf['score'], time.time() - start_time))

        start_time = time.time()
        # 2nd phase: train the fine tuned best classifier on the whole train dataset (no folds)
        estimator = pt.trainClassifier(fX, ytrain, estimator)
        print("Finished training model on dataset; {} sec.".format(time.time() - start_time))

        start_time = time.time()
        fX = f.build(Xtest)
        print("Build features from test data in {} sec".format(time.time() - start_time))

        start_time = time.time()
        # 3th phase: test the fine tuned best classifier on the test dataset
        acc, pre, rec, f1 = pt.testClassifier(fX, ytest, estimator)
        print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (sec)")
        print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(
            best_clf['classifier'], acc, pre, rec, f1, time.time() - start_time))

        print("The whole process took {} sec.".format(time.time() - tot_time))

    def exec_classifiers(self):
        """Train and evaluate selected ML algorithms with custom hyper-parameters on dataset.
        """
        f = Features()
        pt = param_tuning.ParamTuning()

        start_time = time.time()
        Xtrain, Xtest, ytrain, ytest = self._load_and_split_data()
        print("Loaded train/test datasets in {} sec.".format(time.time() - start_time))

        fX_train = f.build(Xtrain)
        fX_test = f.build(Xtest)
        print("Build features from train/test data in {} sec".format(time.time() - start_time))

        for clf in config.MLConf.clf_custom_params:
            print('Method {}'.format(clf))
            print('=======', end='')
            print(len(clf) * '=')

            tot_time = time.time(); start_time = time.time()
            # 1st phase: train each classifier on the whole train dataset (no folds)
            estimator = pt.clf_names[clf][0](**config.MLConf.clf_custom_params[clf])
            estimator = pt.trainClassifier(fX_train, ytrain, estimator)
            print("Finished training model on dataset; {} sec.".format(time.time() - start_time))

            start_time = time.time()
            # 2nd phase: test each classifier on the test dataset
            acc, pre, rec, f1 = pt.testClassifier(fX_test, ytest, estimator)
            print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (sec)")
            print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(
                clf, acc, pre, rec, f1, time.time() - start_time))

            print("The whole process took {} sec.\n".format(time.time() - tot_time))

    def _load_and_split_data(self):
        data_df = gpd.read_file(getRelativePathtoWorking(config.dataset))
        dian_df = gpd.read_file(getRelativePathtoWorking(config.dian))
        dian_df.rename(columns={"geometry": "dian_geom"}, inplace=True)
        data_df = data_df.merge(
            dian_df[['id', 'unique_id', 'area_doc_1', 'worktype', 'dian_geom']],
            left_on=['dian_id'], right_on=['id'], how='left'
        )

        X = data_df.drop('status', axis=1)
        y = data_df['status']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_split_thres, random_state=config.seed_no, stratify=y
        )
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        return X_train, X_test, y_train, y_test
