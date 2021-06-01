|MIT|

=====

#########################
LGM-PolygonClassification
#########################
A python library for effectively classifying land parcel polygons with respect to their provenance information.

===============================
About LGM-PolygonClassification
===============================
LGM-PolygonClassification is a python library that implements a full Machine Learning workflow for training
classification algorithms on annotated datasets that contain pairs of matched polygons each one of which belongs to a
distinct polygon variant. LGM-PolygonClassification implements a series of training features by taking into
consideration the individual characteristics of each polygon as well as including information about the geospatial
relationship between each matched pair. Further, it encapsulates grid-search and cross-validation functionality,
based on the `scikit-learn <https://scikit-learn.org/>`_ toolkit, assessing as series of classification models and
parameterizations, in order to find the most fitting model for the data at hand. Indicatively, we
succeed a 98.44% accuracy with the Gradient Boosting Trees classifier (see `References`_).

The source code was tested using Python 3 (>=3.7) and Scikit-Learn 0.23.1 on a Linux server.

Dependencies
------------
* click==7.1.2
* fiona==1.8.18
* geopandas==0.9.0
* numpy==1.20.2
* pandas==1.2.3
* scikit-learn==0.23.1
* scipy==1.6.2
* shapely==1.7.1
* tabulate==0.8.9
* xgboost==1.3.3

Setup procedure
---------------
Download the latest version from the `GitHub repository <https://github.com/LinkGeoML/LGM-PolygonClassification.git>`_,
change to the main directory and run:

.. code-block:: bash

   pip install -r pip_requirements.txt

It should install all required `dependencies`_ automatically.

Usage
------
The input dataset need to be in CSV format. Specifically, a valid dataset should have at least the following
fields/columns:

* The geometry of the initial, land allocated polygon.
* The geometry of final polygon.
* The ORI\_TYPE label, e.g., {1, 4}, that denotes the dominant provenance of final polygon, i.e., land parcel.

The library implements the following distinct processes:

#. Features extraction
    The `build <https://linkgeoml.github.io/LGM-PolygonClassification/features.html#polygon_classification.features.
    Features>`_ function constructs a set of training features to use within classifiers for toponym interlinking.

#. Algorithm and model selection
    The functionality of the
    `fineTuneClassifiers <https://linkgeoml.github.io/LGM-PolygonClassification/tuning.html#polygon_classification.
    param_tuning.ParamTuning.fineTuneClassifiers>`_ function is twofold.
    Firstly, it chooses among a list of supported machine learning algorithms the one that achieves the highest average
    accuracy score on the examined dataset. Secondly, it searches for the best model, i.e., the best hyper-parameters
    for the best identified algorithm in the first step.

#. Model training
    The `trainClassifier <https://linkgeoml.github.io/LGM-PolygonClassification/tuning.html#polygon_classification.
    param_tuning.ParamTuning.trainClassifier>`_ trains the best selected model on previous
    process, i.e., an ML algorithm with tuned hyperparameters that best fits data, on the whole train dataset, without
    splitting it in folds.

#. Model deployment
    The `testClassifier <https://linkgeoml.github.io/LGM-PolygonClassification/tuning.html#polygon_classification.
    param_tuning.ParamTuning.testClassifier>`_ applies the trained model on new untested data.

A complete pipeline of the above processes, i.e., features extraction, training and evaluating state-of-the-art
classifiers, for polygon classification, i.e., provenance recommendation of a land parcel, can be executed with the
following command:

.. code-block:: bash

    $ python -m polygon_classification.cli run --train_dataset <path/to/train-dataset>
    --test_dataset <path/to/test-dataset>

Additionally, *help* is available on the command line interface (*CLI*). Enter the following to list all supported
commands or options for a given command with a short description.

.. code-block:: bash

    $ python -m polygon_classification.cli –h

    Usage: cli.py [OPTIONS] COMMAND [ARGS]...

    Options:
      -h, --help  Show this message and exit.

    Commands:
      evaluate  evaluate the effectiveness of the proposed methods
      run       A complete process of distinct steps in figuring out the best ML algorithm with optimal hyperparameters...
      train     tune various classifiers and select the best hyper-parameters on a train dataset

Documentation
-------------
Source code documentation is available from `linkgeoml.github.io`__.

__ https://linkgeoml.github.io/LGM-PolygonClassification/

References
----------
* V. Kaffes et al. Determining the provenance of land parcel polygons via machine learning. SSDBM ’20.

License
-------
LGM-PolygonClassification is available under the `MIT <https://opensource.org/licenses/MIT>`_ License.

..
    .. |Documentation Status| image:: https://readthedocs.org/projects/coala/badge/?version=latest
       :target: https://linkgeoml.github.io/LGM-Interlinking/

.. |MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
