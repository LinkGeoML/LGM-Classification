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

from sklearn.cross_validation import train_test_split

import datetime

import config

import glob
import os

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
	for i in range(0, X_test.shape[0]):
		top_k_classes = best_k_probs[:config.initialConfig.k_error]
		if y_test[i] in top_k_classes:
			count += 1
	
	top_k_error = float(count) / float(X_test.shape[0])
	#print("top_k_error: {0}".format(top_k_error))
	
	return top_k_error, baseline_accuracy, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro')

def fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test):
	
	scores = ['accuracy']
	
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

def tuned_parameters_5_fold(poi_ids, conn, args):
	
	# Shuffle ids
	#print(poi_ids[config.initialConfig.poi_id])
	
	poi_ids = poi_ids[config.initialConfig.poi_id]
	random.shuffle(poi_ids)
	
	#poi_ids = np.asarray(poi_ids)
	
	poi_ids = list(poi_ids)
			
	clf_names_not_tuned = ["Naive Bayes", "MLP", "Gaussian Process", "QDA", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
	baseline_scores = []
	top_k_errors = []
	for item in clf_scores_dict:
		clf_scores_dict[item] = [[], [], []]
	report_data = []
	hyperparams_data = []

	train_ids, test_ids = train_test_split(poi_ids, test_size = 0.2)
	
	#print(train_ids, test_ids)
		
	# get train and test sets
	X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, train_ids, test_ids)
		
	most_common_classes = find_10_most_common_classes_train(y_train)
	
	# read clf name from csv
	row = {}
	
	if args['best_clf'] in clf_names_not_tuned:
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
		clf = fine_tune_parameters_given_clf(args['best_clf'], X_train, y_train, X_test, y_test)
	
	
	"""
	#score = clf.score(X_test, y_test)
	top_k_error, baseline_accuracy, accuracy, f1_score_micro, f1_score_macro = get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf)
	clf_scores_dict[clf_name][0].append(accuracy)
	clf_scores_dict[clf_name][1].append(f1_score_micro)
	clf_scores_dict[clf_name][2].append(f1_score_macro)
	row['Fold'] = count
	row['Classifier'] = clf_name
	row['Baseline Accuracy'] = baseline_accuracy
	row['Top-k Accuracy'] = top_k_error
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
	top_k_errors.append(top_k_error)
		
	df = pd.DataFrame.from_dict(report_data)
	
	
	if args['results_file_name'] is not None:
		filename = args['results_file_name'] + '_' + str(datetime.datetime.now()) + '.csv'
	else:
		filename = 'classifier_finetuning_report_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.csv'
	df.to_csv(filename, index = False)
	"""
	
	hyperparams_data = clf.best_params_
	#print(hyperparams_data)
	df2 = pd.DataFrame.from_dict([hyperparams_data])
	
	#print(df2)
	if args['best_hyperparameter_file_name'] is not None:
		filename = args['best_hyperparameter_file_name'] + '_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.csv'
	else:
		filename = 'best_hyperparameters_' + str(args['level']) + '_' + str(datetime.datetime.now()) + '.csv'
	df2.to_csv(filename, index = False)
	
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
	#ap.add_argument("-results_file_name", "--results_file_name", required=False,
	#	help="desired name of best hyperparameter file")
	ap.add_argument("-best_hyperparameter_file_name", "--best_hyperparameter_file_name", required=False,
		help="desired name of best hyperparameter file")
	ap.add_argument("-best_clf_file_name", "--best_clf_file_name", required=False,
		help="name of file containing the best classifier found in step 1")

	args = vars(ap.parse_args())
	
	if args['pois_tbl_name'] is not None:
		print(args['pois_tbl_name'])
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	"""
	if args['best_clf_file_name'] is not None:
		with open(args['best_clf_file_name']) as f:
			args['best_clf'] = f.readline()
	else:
		list_of_files = glob.glob('best_clf_*')
		latest_file = max(list_of_files, key=os.path.getctime)
		with open(latest_file) as f:
			args['best_clf'] = f.readline()
			
	args['best_clf'] = args['best_clf'].rstrip()
	"""
	if args['best_clf_file_name'] is not None:
		#with open(args['best_clf_file_name']) as f:
		#	args['best_clf'] = f.readline()
		input_file = csv.DictReader(open(args['best_clf_file_name']))
		with open(input_file, 'r') as csv_file:
			reader = csv.reader(csv_file)
			count = 0
			for row in reader:
				if count == 1:
					args['best_clf'] = row[0]
				count += 1
	else:
		list_of_files = glob.glob('best_clf_*')
		input_file = max(list_of_files, key=os.path.getctime)
		with open(input_file, 'r') as csv_file:
			reader = csv.reader(csv_file)
			count = 0
			for row in reader:
				if count == 1:
					args['best_clf'] = row[0]
				count += 1
			
	#print(args['best_clf'])
	
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
