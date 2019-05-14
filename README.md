# LGM-Classification
A python library for accurate classification of Points of Interest (POIs) into categories.

## About LGM-Classification
LGM-Classification is a python library implementing a full Machine LEarning workflow for training classification algorithms on annotated POI datasets and producing models for the accurate classification of Points of Interest (POIs) into categories. LGM-Classification implements a series of training features, regarding the properties POIs and their relations with neighboring POIs. Further, it it encapsulates grid-search and cross-validation functionality, based on the [scikit](https://scikit-learn.org/) toolkit, assessing as series of classification models and parameterizations, in order to find the most fitting model for the data at hand.

## Dependencies
* python 3
* numpy
* pandas
* sklearn
* geopandas
* nltk
* matplotlib
* psycopg2
* osmnx
* shapely
* argparse

## Instructions
In order for the library to function the user must provide it with a .csv file containing a collection of POI details. More specifically, the .csv file must at least contain specific poi information that correspond to the following:

- a column that corresponds to the unique ID of each poi
- the name of each POI
- a collection of class names in which each POI belongs
- the latitude of the POI
- the longitude of the POI

**Algorithm evaluation/selection**: consists of an exhaustive comparison between several classification algorithms that are available in the scikit-learn library. Its purpose is to
compare the performance of every algorithm-hyperparameter configuration in a nested cross-validation scheme and produce the best candidate-algorithm for further usage. More specifically this step outputs three files: 

* a file consisting of the algorithm and parameters space that was searched, 
* a file containing the results per cross-validation fold and their averages and 
* a file containing the name of the best model.

You can execute this step as follows: ```python find_best_clf.py -pois_csv_name <csv containing poi information> -results_file_name <desired name of the csv to contain the metric results per fold> -hyperparameter_file_name <desired name of the file to contain the hyperparameter space that was searched>```.

The last two arguments are optional and their values are defaulted to:
* classification_report_*category level*_*timestamp*, and 
* hyperparameters_per_fold_*category level*_*timestamp*

correspondingly

**Algorithm tuning**: The purpose of this step is to further tune the specific algorithm that was chosen in step 1 by comparing its performance while altering the hyperparameters with which it is being configured. This step outputs the hyperparameter selection corresponding to the best model.

You can execute this step as follows: ```python finetune_best_clf.py -pois_csv_name <csv containing poi information> -best_hyperparameter_file_name <desired name of the file to contain the best hyperparameters that were selected for the best algorithm of step 1> -best_clf_file_name <file containing the name of the best classifier>```.

All arguments except pois_csv_name are optional and their values are defaulted to:

* best_hyperparameters_*category level*_*timestamp*.csv
* the latest file with the *best_clf_* prefix

**Model training on a specific training set**: This step handles the training of the final model on an entire dataset, so that it can be used in future cases. It outputs a pickle file in which the model is stored.

You can execute this step as follows: ```python export_best_model.py -pois_csv_name <csv containing poi information> -best_hyperparameter_file_name <csv containing best hyperparameter configuration for the classifier -best_clf_file_name <file containing the name of the best classifier> -trained_model_file_name <name of file where model must be exported>```.

All arguments except pois_csv_name are optional and their values are defaulted to:

* the latest file with the *best_hyperparameters_* prefix
* the latest file with the best_clf_* prefix
* trained_model_*level*_*timestamp*.pkl

correspondingly.

**Predictions on novel data**: This step can be executed as: ```python export_predictions.py -pois_csv <csv containing poi information> -k <desired number of predicted categories per poi> -results_file_name <desired name of the output csv> -trained_model_file_name <pickle file containing an already trained model>```

The output .csv file will contain the k most probable predictions regarding the category of each POI. If no arguments for k and output_csv are given, their values are defaulted to:
* k = 5, 
* output_csv = top_k_predictions_*timestamp*.csv and 
* trained_model_file_name = *name of the latest produced pickle file in the working directory*.

## Use case: Yelp dataset

If we want to test our trained model's performance on another dataset (e.g. the Yelp dataset) we can run the following sequence of commands:

```python find_best_clf.py -pois_csv_name pois_data.csv```

in order to find the best overall classification algorithm,

```python finetune_best_clf.py -pois_csv_name pois_data.csv```

in order to find the best hyperparameters setting for it,

```python export_best_model.py -pois_csv_name pois_data.csv```

in order to train the model on the whole poi dataset and finally

```python other_dataset_classification.py -pois_csv_name yelp_LasVegas.csv```

to finally take the category predictions for each poi.

In order for the last step to be executed there are some alterations that need to be made in the config.py file. More specifically we must change:

* poi_id = "poi_id" to poi_id = "id"
* class_codes = ["theme", "class_name", "subclass_n"] to class_codes = ["class_name"]
* original_SRID = 'epsg:2100' to original_SRID = 'epsg:3857'
* level = [1, 2] to level = [1]
* osmnx_placename = "Marousi, Athens, Greece" to osmnx_placename = "Las Vegas"
