# -*- coding: utf-8 -*-
# Author: vkaff
# E-mail: vkaffes@imis.athena-innovation.gr

import time
from sklearn.model_selection import train_test_split
from joblib import dump, load
import numpy as np
import pandas as pd
# We'll use this library to make the display pretty
from tabulate import tabulate
import os

from polygon_classification import param_tuning, config
from polygon_classification.features import Features
from polygon_classification.helpers import StaticValues


class StrategyEvaluator:
    """
    This class implements the pipeline for various strategies.
    """
    max_features_toshow = 10

    def __init__(self):
        pass

    def hyperparamTuning(self, dataset, classifiers):
        """A complete process of distinct steps in figuring out the best ML algorithm with best hyperparameters to
        polygon classification problem.
        """
        pt = param_tuning.ParamTuning()
        f = Features()

        tot_time = time.time(); start_time = time.time()
        Xtrain, Xtest, ytrain, ytest = self._load_and_split_data(dataset)
        print("Loaded train/test datasets in {} sec.".format(time.time() - start_time))

        fX = f.build(Xtrain)
        print("Build features from train data in {} sec.".format(time.time() - start_time))

        start_time = time.time()
        # 1st phase: find and fine tune the best classifier from a list of candidate ones
        best_clf = pt.fineTuneClassifiers(fX, ytrain, classifiers)
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
        res = pt.testClassifier(fX, ytest, estimator)
        self._print_stats(best_clf['clf_name'], res['metrics'], res['feature_imp'], start_time)

        print("The whole process took {} sec.".format(time.time() - tot_time))

    def exec_classifiers(self, dataset):
        """Train and evaluate selected ML algorithms with custom hyper-parameters on dataset.
        """
        f = Features()
        pt = param_tuning.ParamTuning()

        start_time = time.time()
        Xtrain, Xtest, ytrain, ytest = self._load_and_split_data(dataset)
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
            # estimator = pt.clf_names[clf][0](**config.MLConf.clf_custom_params[clf])
            estimator = pt.clf_names[clf][0](random_state=config.seed_no)
            estimator.set_params(**config.MLConf.clf_custom_params[clf])
            estimator = pt.trainClassifier(fX_train, ytrain, estimator)

            print("Finished training model on dataset; {} sec.".format(time.time() - start_time))

            start_time = time.time()
            # 2nd phase: test each classifier on the test dataset
            res = pt.testClassifier(fX_test, ytest, estimator)
            self._print_stats(clf, res['metrics'], res['feature_imp'], start_time)
            # if not os.path.exists('output'):
            #     os.makedirs('output')
            # np.savetxt(f'output/{clf}_default_stats.csv', res['metrics']['stats'], fmt="%u")

            print("The whole process took {} sec.\n".format(time.time() - tot_time))

    def train(self, dataset, classifiers):
        """A complete process of distinct steps in figuring out the best ML algorithm with best hyperparameters to
        polygon classification problem.
        """
        pt = param_tuning.ParamTuning()
        f = Features()

        tot_time = time.time(); start_time = time.time()
        data_df = pd.read_csv(dataset)
        ytrain = data_df['status']
        Xtrain = data_df.drop('status', axis=1)
        print("Loaded train dataset in {} sec.".format(time.time() - start_time))

        fX = f.build(Xtrain)
        print("Build features from train data in {} sec.".format(time.time() - start_time))

        start_time = time.time()
        # 1st phase: find and fine tune the best classifier from a list of candidate ones
        best_clf = pt.fineTuneClassifiers(fX, ytrain, classifiers)
        estimator = best_clf['estimator']
        print("Best hyperparams for {}, {}, with score {}; {} sec.".format(
            best_clf['hyperparams'], best_clf['clf_name'], best_clf['score'], time.time() - start_time))

        estimator = pt.trainClassifier(fX, ytrain, estimator)
        os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
        dump(estimator, os.path.join(os.getcwd(), 'models', best_clf['clf_name'] + '_model.joblib'))

        print("The whole process took {} sec.".format(time.time() - tot_time))

    def evaluate(self, dataset, classifier):
        """A complete process of distinct steps in figuring out the best ML algorithm with best hyperparameters to
        polygon classification problem.
        """
        pt = param_tuning.ParamTuning()
        f = Features()

        tot_time = time.time(); start_time = time.time()
        # Xtrain, Xtest, ytrain, ytest = self._load_and_split_data(dataset)
        data_df = pd.read_csv(dataset)
        ytest = data_df['status']
        Xtest = data_df.drop('status', axis=1)
        print("Loaded test dataset in {} sec.".format(time.time() - start_time))

        start_time = time.time()
        fX = f.build(Xtest)
        print("Build features from test data in {} sec".format(time.time() - start_time))

        start_time = time.time()
        # 3th phase: test the fine tuned best classifier on the test dataset
        estimator = load(os.path.join(os.getcwd(), 'models', classifier + '_model.joblib'))
        res = pt.testClassifier(fX, ytest, estimator, True)
        self._print_stats(classifier, res['metrics'], res['feature_imp'], start_time)

        Xtest.reset_index(inplace=True)
        Xtest = pd.concat([Xtest, pd.DataFrame(res['proba'], columns=['none_origin_pred', 'dian_origin_pred'])], axis=1)
        os.makedirs('output', exist_ok=True)
        Xtest[['pst_geom', 'dian_geom', 'none_origin_pred', 'dian_origin_pred']].to_csv(
            os.path.join('output', 'predictions.csv'), index=False)

        print("The whole process took {} sec.".format(time.time() - tot_time))

    def _print_stats(self, clf, params, fimp, tt):
        print("| Method\t\t& Accuracy\t& Precision\t& Recall\t& F1-Score\t& Time (sec)")
        print("||{0}\t& {1}\t& {2}\t& {3}\t& {4}\t& {5}".format(
            clf, params['accuracy'], params['precision'], params['recall'], params['f1_score'],
            time.time() - tt))

        if fimp is not None:
            importances = np.ma.masked_equal(fimp, 0.0)
            if importances.mask is np.ma.nomask: importances.mask = np.zeros(importances.shape, dtype=bool)

            indices = np.argsort(importances.compressed())[::-1][
                      :min(importances.compressed().shape[0], self.max_features_toshow)]
            headers = ["name", "score"]

            fcols = StaticValues.featureCols if config.MLConf.extra_features is False \
                else StaticValues.featureCols + StaticValues.extra_featureCols
            print(tabulate(zip(
                np.asarray(fcols, object)[~importances.mask][indices], importances.compressed()[indices]
            ), headers, tablefmt="simple"))

        print()

    def _load_and_split_data(self, dataset):
        data_df = pd.read_csv(dataset)
        # dian_df = gpd.read_file(getRelativePathtoWorking(config.dian))
        # dian_df.rename(columns={"geometry": "dian_geom"}, inplace=True)
        # data_df = data_df.merge(
        #     dian_df[['id', 'unique_id', 'area_doc_1', 'worktype', 'dian_geom']],
        #     left_on=['dian_id'], right_on=['id'], how='left'
        # )

        X = data_df.drop('status', axis=1)
        y = data_df['status']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_split_thres, random_state=config.seed_no, stratify=y, shuffle=True
        )
        # pd.concat([X_train, y_train], axis=1).to_csv('data/train_dataset.csv', index=False)
        # pd.concat([X_test, y_test], axis=1).to_csv('data/test_dataset.csv', index=False)

        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        return X_train, X_test, y_train, y_test
