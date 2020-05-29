# LGM-Classification
A python library for accurate classification of Points of Interest (POIs) into categories.

## About LGM-Classification
LGM-Classification is a python library implementing a full Machine Learning workflow for training classification algorithms on annotated POI datasets and producing models for the accurate classification of Points of Interest (POIs) into categories. LGM-Classification implements a series of training features, regarding the properties POIs and their relations with neighboring POIs. Further, it it encapsulates grid-search and cross-validation functionality, based on the [scikit](https://scikit-learn.org/) toolkit, assessing as series of classification models and parameterizations, in order to find the most fitting model for the data at hand.

## Description
The module consists of the following steps:

1. **Features extraction**

   This step takes into account the features hyperparameters given in [config.py](./src/config.py) and creates a collection of X_train and X_test feature pairs grouped by folds and ready to be utilized by machine learning algorithms in the next steps.
2. **Algorithm selection**

   A list of given classifiers given in [config.py](./src/config.py) are evaluated in a nested cross-validation scheme in order to find which performs the best on the features sets created in the previous step.
3. **Model selection**

   Given a selected classifier, this step tries to find the best configuration both in terms of the classifier's and feature set hyperparameters.
4. **Model training**

   Utilizing the knowledge from the previous step, a model is trained on the whole available pois using the optimal configuration. This model is then saved to disk for later usage.
5. **Model deployment**

   This step loads the optimal model from disk and uses it in order to classify a set of unseen, unlabeled, test pois. Classification results come in a form of a top k predictions list for each of the pois along with a score (suggesting the model's confidence about each prediction).

## Usage
The execution of the project starts with the **Features extraction** step initializing the pipeline's root folder which the following steps will refer to in order to output their results. Each step can be executed as follows:

1. **Features extraction**

   ```python features_extraction.py -poi_fpath <poi_fpath>```
   
   where ```<poi_fpath>``` is the path to the file containing the train pois.
2. **Algorithm selection**

   ```python algorithm_selection.py -experiment_path <exp_path>```
   
   where ```<exp_path>``` is the path to the folder created from the first step.
3. **Model selection**

   ```python model_selection.py -classifier <clf_name> -experiment_path <exp_path>```
   
   where ```<clf_name>``` is the classifier's name to be optimized in order to build the model and ```<exp_path>``` same as before.
4. **Model training**

   ```python model_training.py -experiment_path <exp_path>```
   
   where ```<exp_path>``` same as before.
5. **Model deployment**

   ```python model_deployment.py -experiment_path <exp_path> -poi_fpath <poi_fpath>```
   
   where ```<exp_path>``` same as before and ```<poi_fpath>``` is the path to the file containing the test pois.

## Documentation
Source code documentation is available from [linkgeoml.github.io](https://linkgeoml.github.io/LGM-Classification/).
