# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import time
from sklearn.model_selection import train_test_split
import geopandas as gpd
import numpy as np
from itertools import chain

from src import param_tuning
from src import config
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
        toponym interlinking problem.

        :param train_data: Relative path to the train dataset.
        :type train_data: str
        :param test_data: Relative path to the test dataset.
        :type test_data: str
        """
        pt = param_tuning.ParamTuning()
        f = Features()

        tot_time = time.time(); start_time = time.time()
        Xtrain, Xtest, ytrain, ytest = self._load_and_split_data()
        fX = f.build(Xtrain)
        print("Loaded train dataset and build features in {} sec.".format(time.time() - start_time))

        start_time = time.time()
        # 1st phase: find out best classifier from a list of candidate ones
        # best_clf = pt.getBestClassifier(fX, ytrain)
        # print("Best classifier is {} with score {}; {} sec.".format(
        #     best_clf['classifier'], best_clf['accuracy'], time.time() - start_time))
        #
        # start_time = time.time()
        # #  2nd phase: fine tune the best classifier in previous step
        # estimator, params, score = pt.fineTuneClassifier(fX, ytrain, best_clf)
        best_clf, estimator = pt.fineTuneClassifiers(fX, ytrain)
        print("Best hyperparams, {}, with score {}; {} sec.".format(
            best_clf['hyperparams'], best_clf['score'], time.time() - start_time))

        start_time = time.time()
        # 3nd phase: train the fine tuned best classifier on the whole train dataset (no folds)
        estimator = pt.trainClassifier(fX, ytrain, estimator)
        print("Finished training model on the dataset; {} sec.".format(time.time() - start_time))

        start_time = time.time()
        fX = f.build(Xtest)
        print("Loaded test dataset and build features; {} sec".format(time.time() - start_time))

        start_time = time.time()
        # 4th phase: test the fine tuned best classifier on the test dataset
        acc, pre, rec, f1 = pt.testClassifier(fX, ytest, estimator)
        print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (sec)")
        print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(
            best_clf['classifier'], acc, pre, rec, f1, time.time() - start_time))

        print("The whole process took {} sec.".format(time.time() - tot_time))

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
