#!/usr/bin/python
import numpy as np

class initialConfig:
	
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
		'min_samples_split': list(np.linspace(0.1,1,10)),
		'min_samples_leaf': list(np.linspace(0.1,0.5,5)),
                  'max_features': [i for i in range(1, 10)]}
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
	
	poi_id = "poi_id"
	#poi_id = "id"
	name = "name"
	#class_codes = ["class_name"]
	class_codes = ["theme", "class_name", "subclass_n"]
	x = "x"
	y = "y"
	
	# The following parameter corresponds to the original SRID of the
	# poi dataset that is being fed to the experiments.
	
	original_SRID = 'epsg:2100'
	#original_SRID = 'epsg:3857'
	
	# The following parameters correspond to the various options
	# regarding the geospacial feature extraction phase.
	
	# This parameter refers to the radius that corresponds to the
	# criterion by which we decide whether a poi is adjacent to
	# another.
	threshold_distance_neighbor_pois = 200.0
	
	# This parameter refers to the radius that corresponds to the
	# criterion by which we decide whether a poi is adjacent to
	# a street.
	threshold_distance_neighbor_pois_roads = 300.0
	
	# The following parameters correspond to the various options
	# regarding the textual feature extraction phase.
	
	# This parameter refers to the percentage of term n-grams to be
	# utilized in the feature extraction phase. E.g. when it is
	# set to 0.1, only the top 10% term n-grams will be used.
	top_k_terms_percentage = 0.1
	
	# This parameter refers to the percentage of character n-grams to be
	# utilized in the feature extraction phase. E.g. when it is
	# set to 0.1, only the top 10% character n-grams will be used.
	top_k_character_ngrams_percentage = 0.1
	
	# These parameters refer to the size of the character n-grams
	# and term n-grams respectively.
	character_n_gram_size = 3
	term_n_gram_size = 2
	
	# This parameter refers to the category levels to be predicted.
	# If level is equal to None, the experiments will be run for
	# all category levels, one at a time.
	#level = [1, 2]
	level = [2]
	#level = [1]
	
	# This parameter refers to the desired numbers of the top k most probable
	# predictions to be taken into account for computing the top-k accuracies.
	k_error = [5, 10]
	
	# Parameters regarding the osmnx toolkit
	
	# if this parameter is set to True then the coordinates of the bounding box must be
	# given in the osmnx_bbox_coordinates parameter as a list. This bounding box geospacially
	# must geospacially correspond to the place that we want its street data to be included in 
	# the feature extraction process.
	osmnx_bbox = False
	osmnx_bbox_coordinates = []
	
	# If osmnx_bbox is set to False then osmnx_placename must be set to a string
	# that corresponds to the name of the place that we want its street data to
	# be included in the feature extraction process.
	osmnx_placename = "Marousi, Athens, Greece"
	#osmnx_placename = "Las Vegas, Nevada"
