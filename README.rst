#########################
LGM-PolygonClassification
#########################

A python library for accurate classification of polygon types.

===============================
About LGM-PolygonClassification
===============================
LGM-PolygonClassification is a python library that implements a full Machine Learning workflow for training classification algorithms on annotated datasets that contain pairs of matched polygons each one of which belongs to a distinct polygon variant. LGM-PolygonClassification implements a series of training features by taking into consideration the individual characteristics of each polygon as well as including information about the geospatial relationship between each matched pair. Further, it encapsulates grid-search and cross-validation functionality, based on the [scikit](https://scikit-learn.org/) toolkit, assessing as series of classification models and parameterizations, in order to find the most fitting model for the data at hand.

Dependencies
------------
* python 3


Instructions
------------

.. In order for the library to function the user must provide it with a .csv file containing a collection of matched polygon pairs. The first column must contain the polygon shapely geometries (in string form) that belong to the first polygon class, while the second column must contain their matched counterparts that belong to the second polygon class. The process of polygon matching is also supported by the library, provided that a pair of shapefiles containing polygon information (one for each polygon class) is available.

.. **Polygon matching**: the process of matching polygons can be executing by calling the match_polygons.py script as follows:
.. ```python match_polygons.py -dian_shapefile_name <shapefile that corresponds to the first class polygons> -pst_shapefile_name <shapefile that corresponds to the second class polygons>```.

.. **Algorithm evaluation/selection**: consists of an exhaustive comparison between several classification algorithms that are available in the scikit-learn library. Its purpose is to
.. compare the performance of every algorithm-hyperparameter configuration in a nested cross-validation scheme and produce the best candidate-algorithm for further usage. More specifically this step outputs three files:

.. * a file consisting of the algorithm and parameters space that was searched,
.. * a file containing the results per cross-validation fold and their averages and
.. * a file containing the name of the best model.

.. You can execute this step as follows: ```python find_best_clf.py -polygon_file_name <csv containing polygon pairs information> -results_file_name <desired name of the csv to contain the metric results per fold> -hyperparameter_file_name <desired name of the file to contain the hyperparameter space that was searched>```.

.. The last two arguments are optional and their values are defaulted to:

.. * classification_report_*timestamp*, and
.. * hyperparameters_per_fold_*timestamp*

.. correspondingly

.. **Algorithm tuning**: The purpose of this step is to further tune the specific algorithm that was chosen in step 1 by comparing its performance while altering the hyperparameters with which it is being configured. This step outputs the hyperparameter selection corresponding to the best model.

.. You can execute this step as follows: ```python finetune_best_clf.py -polygon_file_name <csv containing polygon pairs information> -best_hyperparameter_file_name <desired name of the file to contain the best hyperparameters that were selected for the best algorithm of step 1> -best_clf_file_name <file containing the name of the best classifier>```.

.. All arguments except pois_csv_name are optional and their values are defaulted to:

.. * best_hyperparameters_*category level*_*timestamp*.csv
.. * the latest file with the *best_clf_* prefix

.. **Model training on a specific training set**: This step handles the training of the final model on an entire dataset, so that it can be used in future cases. It outputs a pickle file in which the model is stored.

.. You can execute this step as follows: ```python export_best_model.py -polygon_file_name <csv containing polygon pairs information> -best_hyperparameter_file_name <csv containing best hyperparameter configuration for the classifier -best_clf_file_name <file containing the name of the best classifier> -trained_model_file_name <name of file where model must be exported>```.

.. All arguments except pois_csv_name are optional and their values are defaulted to:

.. * the latest file with the *best_hyperparameters_* prefix
.. * the latest file with the best_clf_* prefix
.. * trained_model_*level*_*timestamp*.pkl

.. correspondingly.

.. **Predictions on novel data**: This step can be executed as: ```python export_predictions.py -polygon_file_name <csv containing polygon pairs information> -results_file_name <desired name of the output csv> -trained_model_file_name <pickle file containing an already trained model>```

.. The output .csv file will contain the k most probable predictions regarding the category of each POI. If no arguments for output_csv are given, their values are defaulted to:

.. * output_csv = predictions_*timestamp*.csv and
.. * trained_model_file_name = *name of the latest produced pickle file in the working directory*.

License
-------
LGM-PolygonClassification is available under the MIT License.
