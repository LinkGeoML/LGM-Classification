#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from pois_feature_extraction import *
from textual_feature_extraction import *
from feml import *
import nltk
import itertools
import random

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import datetime

import config

np.random.seed(1234)

def get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf):
	
	top_class_count = 0
	for label in y_test:
		if label == most_common_classes[0]:
			top_class_count += 1
	
	#print("Baseline accuracy: {0}", format(float(top_class_count) / float(y_test.shape[0])))
	baseline_accuracy = float(top_class_count) / float(y_test.shape[0])
	y_pred = clf.predict(X_test)	
	
	#print(X_test.shape)
	probs = clf.predict_proba(X_test)
	#print(probs)
	best_k_probs = np.argsort(probs, axis = 1)
	#print(best_k.shape)
	#print(best_k)
	count = 0
	top_k_errors = []
	for k in config.initialConfig.k_error:
		for i in range(0, X_test.shape[0]):
			top_k_classes = best_k_probs[:k]
			if y_test[i] in top_k_classes:
				count += 1
		
		top_k_error = float(count) / float(X_test.shape[0])
		top_k_errors.append(top_k_error)
	#print("top_k_error: {0}".format(top_k_error))
	
	return top_k_errors, baseline_accuracy, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro')

def fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test):
	
	#scores = ['precision', 'recall']
	scores = ['accuracy']#, 'f1_macro', 'f1_micro']
	
	if clf_name == "SVM":

		tuned_parameters = config.initialConfig.SVM_hyperparameters
		clf = SVC(probability=True)
		
	elif clf_name == "Nearest Neighbors":
		tuned_parameters = config.initialConfig.kNN_hyperparameters
		clf = KNeighborsClassifier()
		
	elif clf_name == "Decision Tree":

		tuned_parameters = config.initialConfig.DecisionTree_hyperparameters
		clf = DecisionTreeClassifier()
		
	elif clf_name == "Random Forest":

		tuned_parameters = config.initialConfig.RandomForest_hyperparameters
		clf = RandomForestClassifier()
	
	"""
	elif clf_name == "AdaBoost":
		tuned_parameters = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": [1, 2]
             }
		clf = AdaBoostClassifier()
	
	elif clf_name == "MLP":
		tuned_parameters = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
		clf = MLPClassifier()
		
	elif clf_name == "Gaussian Process":
		
		clf = GaussianProcessClassifier()
	
	elif clf_name == "QDA":
		tuned_parameters = 
		clf = QuadraticDiscriminantAnalysis()
	"""
	
	print(clf_name)
		
	for score in scores:

		clf = GridSearchCV(clf, tuned_parameters, cv=4,
						   scoring='%s' % score, verbose=0)
		clf.fit(X_train, y_train)
		
	return clf
	
def feature_selection(X_train, X_test, y_train):
		
	from sklearn.feature_selection import VarianceThreshold
	from sklearn.feature_selection import SelectKBest
	from sklearn.feature_selection import chi2
	from sklearn.feature_selection import SelectFromModel
	from sklearn.feature_selection import RFECV
	from sklearn.svm import SVC
	from sklearn.svm import LinearSVC
	from sklearn.model_selection import StratifiedKFold
	
	print("Before feature selection - X_train:{0}, X_test:{1}".format(X_train.shape, X_test.shape))	
	
	# Variance Threshold feature selection
	sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	X_train = sel.fit_transform(X_train)
	#print(X_train.shape)
	feature_mask = sel.get_support()
	X_test_new = np.zeros((X_test.shape[0], X_train.shape[1]))
	#print(feature_mask)
	for i in range(X_test.shape[0]):
		count = 0
		for j in range(X_test.shape[1]):
			if feature_mask[j] == True:
				X_test_new[i][count] = X_test[i][j]
				count += 1
	
	
	# Univariate feature selection
	sel = SelectKBest(chi2, k=2)
	X_train = sel.fit_transform(X_train, y_train)
	feature_mask = sel.get_support()
	X_test_new = np.zeros((X_test.shape[0], X_train.shape[1]))
	for i in range(X_test.shape[0]):
		count = 0
		for j in range(X_test.shape[1]):
			if feature_mask[j] == True:
				X_test_new[i][count] = X_test[i][j]
				count += 1
		
	# Recursive cross-validated feature elimination
	svc = SVC(kernel="linear")
	sel = RFECV(estimator=svc, step=1, cv=StratifiedKFold(5),
				  scoring='accuracy')
	sel.fit(X_train, y_train)
	feature_mask = sel.get_support()
	number_of_features = list(feature_mask).count(True)
	X_train_new = np.zeros((X_train.shape[0], number_of_features))
	X_test_new = np.zeros((X_test.shape[0], number_of_features))
	for i in range(X_test.shape[0]):
		count = 0
		for j in range(X_test.shape[1]):
			if feature_mask[j] == True:
				X_test_new[i][count] = X_test[i][j]
				count += 1
				
	for i in range(X_train.shape[0]):
		count = 0
		for j in range(X_train.shape[1]):
			if feature_mask[j] == True:
				X_train_new[i][count] = X_train[i][j]
				count += 1
	
	# L1-norm based feature selection
	svc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
	sel = SelectFromModel(svc, prefit=True)
	X_train = sel.transform(X_train)
	feature_mask = sel.get_support()
	X_test_new = np.zeros((X_test.shape[0], X_train.shape[1]))
	for i in range(X_test.shape[0]):
		count = 0
		for j in range(X_test.shape[1]):
			if feature_mask[j] == True:
				X_test_new[i][count] = X_test[i][j]
				count += 1
	
	X_test = X_test_new
	print("After feature selection - X_train:{0}, X_test:{1}".format(X_train.shape, X_test.shape))
	
	return X_train, X_test
		
def tuned_parameters_5_fold(poi_ids, conn, args):
	
	# Shuffle ids
	poi_ids = poi_ids[config.initialConfig.poi_id]
	random.shuffle(poi_ids)
	
	kf = KFold(n_splits = config.initialConfig.k_fold_parameter)
	
	count = 1
	
	clf_names_not_tuned = ["Naive Bayes", "MLP", "Gaussian Process", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
	baseline_scores = []
	top_k_errors = [[] for _ in range(0, len(config.initialConfig.k_error))]
	for item in clf_scores_dict:
		clf_scores_dict[item] = [[], [], []]
	report_data = []
	hyperparams_data = []
	
	# split data into train, test
	for train_ids, test_ids in kf.split(poi_ids):
		
		train_ids = [train_id + 1 for train_id in train_ids]
		test_ids = [test_id + 1 for test_id in test_ids]
		
		# get train and test sets
		X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, train_ids, test_ids)
		
		#X_train, X_test = feature_selection(X_train, X_test, y_train)
		
		most_common_classes = find_10_most_common_classes_train(y_train)
		
		for clf_name in clf_names:
			row = {}
			#print(clf_name)
			
			if clf_name in clf_names_not_tuned:
				if clf_name == "Naive Bayes":
					clf = GaussianNB()
					clf.fit(X_train, y_train)
				elif clf_name == "MLP":
					clf = MLPClassifier()
					clf.fit(X_train, y_train)
				elif clf_name == "Gaussian Process":
					clf = GaussianProcessClassifier()
					clf.fit(X_train, y_train)
				#elif clf_name == "QDA":
				#	clf = QuadraticDiscriminantAnalysis()
				#	clf.fit(X_train, y_train)
				else:
					clf = AdaBoostClassifier()
					clf.fit(X_train, y_train)
			else:
				clf = fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test)
			
			#score = clf.score(X_test, y_test)
			top_k_error_list, baseline_accuracy, accuracy, f1_score_micro, f1_score_macro = get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf)
			clf_scores_dict[clf_name][0].append(accuracy)
			clf_scores_dict[clf_name][1].append(f1_score_micro)
			clf_scores_dict[clf_name][2].append(f1_score_macro)
			row['Fold'] = count
			row['Classifier'] = clf_name
			row['Baseline Accuracy'] = baseline_accuracy
			i = 0
			for k, top_k_error in zip(config.initialConfig.k_error, top_k_error_list):
				row['Top-{0} Accuracy'.format(k)] = top_k_error
				top_k_errors[i].append(top_k_error)
				i += 1
			row['Accuracy'] = accuracy
			row['F1-Score-Micro'] = f1_score_micro
			row['F1-Score-Macro'] = f1_score_macro
			
			if clf_name not in clf_names_not_tuned:
				hyperparams_row = {}
				hyperparams_row['Fold'] = count
				hyperparams_row['Classifier'] = clf_name
				hyperparams_row['Best Hyperparameters'] = clf.best_params_
				hyperparams_data.append(hyperparams_row)
				
			report_data.append(row)
			baseline_scores.append(baseline_accuracy)
			
		count += 1
		
	for clf_name in clf_names:	
		row = {}
		row['Fold'] = 'average'
		row['Classifier'] = clf_name
		row['Accuracy'] = sum(map(float,clf_scores_dict[clf_name][0])) / 5.0 
		i = 0
		for k, top_k_error in zip(config.initialConfig.k_error, top_k_error_list):
			row['Top-{0} Accuracy'.format(k)] = sum(map(float,top_k_errors[i])) / 5.0
			i += 1
		#row['Top-k Accuracy'] = sum(map(float,top_k_errors)) / 5.0
		row['Baseline Accuracy'] = sum(map(float,baseline_scores)) / 5.0
		row['F1-Score-Micro'] = sum(map(float,clf_scores_dict[clf_name][1])) / 5.0
		row['F1-Score-Macro'] = sum(map(float,clf_scores_dict[clf_name][2])) / 5.0
		report_data.append(row)
		
	df = pd.DataFrame.from_dict(report_data)
	if args['results_file_name'] is not None:
		filename = args['results_file_name'] + '_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.csv'
	else:
		filename = 'classification_report_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.csv'
	df.to_csv(filename, index = False)
	
	best_clf_row = {}
	best_clf_row['best_clf_score'] = 0.0
	best_clf_row['best_clf_name'] = ''
	for index, row in df.iterrows():
		if row['Fold'] == 'average':
			if row['Accuracy'] > best_clf_row['best_clf_score'] :
				best_clf_row['best_clf_score']  = row['Accuracy']
				best_clf_row['best_clf_name'] = row['Classifier']
	df2 = pd.DataFrame.from_dict([best_clf_row])
	filename = 'best_clf_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.csv'
	df2.to_csv(filename, index = False)
	
	df3 = pd.DataFrame.from_dict(hyperparams_data)
	if args['hyperparameter_file_name'] is not None:
		filename = args['hyperparameter_file_name'] + '_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.csv'
	else:
		filename = 'hyperparameters_per_fold_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.csv'
	df3.to_csv(filename, index = False)
	
def write_data_to_csv(conn, args):
	sql = "select {0}.id, {0}.name_u, {0}.theme, {0}.class_name, {0}.subclass_n, {0}.x, {0}.y, {0}.geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	df.to_csv("pois_data.csv", index = False)
	print("ok")
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=False,
		help="name of table containing pois information")
	ap.add_argument("-pois_csv_name", "--pois_csv_name", required=False,
		help="name of csv containing pois information")
	ap.add_argument("-results_file_name", "--results_file_name", required=False,
		help="desired name of output file")
	ap.add_argument("-hyperparameter_file_name", "--hyperparameter_file_name", required=False,
		help="desired name of output file")
	ap.add_argument("-retrain_csv_file_name", "--retrain", required=False,
		help="name of csv containing the dataset you want to retrain the algorithm on")

	args = vars(ap.parse_args())
	
	if args['pois_tbl_name'] is not None:
		print(args['pois_tbl_name'])
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	"""
	args['threshold_pois'] = config.initialConfig.threshold_distance_neighbor_pois
	args['threshold_streets'] = config.initialConfig.threshold_distance_neighbor_pois_roads
	args['k_ngrams'] = config.initialConfig.top_k_character_ngrams_percentage
	args['k_tokens'] = config.initialConfig.top_k_terms_percentage
	args['n'] = config.initialConfig.character_n_gram_size
	args['n_tokens'] = config.initialConfig.term_n_gram_size
	args['level'] = config.initialConfig.level
	"""
	#write_data_to_csv(conn, args)
	
	# get the poi ids
	if config.initialConfig.level == None:
		for level in [1,2]:
			args['level'] = level
			poi_ids = get_poi_ids(conn, args)
			tuned_parameters_5_fold(poi_ids, conn, args)
	else:
		for level in config.initialConfig.level:
			args['level'] = level
			poi_ids = get_poi_ids(conn, args)
			tuned_parameters_5_fold(poi_ids, conn, args)
	
	
if __name__ == "__main__":
   main()
