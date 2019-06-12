#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from pois_feature_extraction_csv import *
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
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
	
	"""
	This function is responsible for measuring the prediction scores
	during the test phase
	
	Arguments
	---------
	X_test: :obj:`numpy array`
		the test set features
	y_test: :obj:`numpy array`
		the test set class labels
	most_common_classes: :obj:`list`
		the 10 most common (most populated) classes
	clf: :obj:`scikit-learn classifier object`
		the classifier object that is used for the predictions
	
	Returns
	-------
	top_k_errors: :obj:`list`
		a list containing the top-k-error measurements
	baseline_accuracy: :obj:`float`
		the baseline accuracy (the most populated class is assigned to every prediction)
	baseline_f_score: :obj:`float`
		same as above, but f_score
	accuracy_score(y_test, y_pred): :obj:`float`
		the accuracy score as computed by scikit-learn
	f1_score(y_test, y_pred, average='weighted'): :obj:`float`
		the weighted f1-score as computed by scikit-learn
	f1_score(y_test, y_pred, average='macro'): :obj:`float`
		the macro f1-score as computed by scikit-learn
	"""
	
	top_class_count = 0
	for label in y_test:
		if label == most_common_classes[0]:
			top_class_count += 1
	
	baseline_accuracy = float(top_class_count) / float(y_test.shape[0])
	y_pred = clf.predict(X_test)	
	
	probs = clf.predict_proba(X_test)
	best_k_probs = np.argsort(probs, axis = 1)
	count = 0
	for i in range(0, X_test.shape[0]):
		top_k_classes = best_k_probs[:config.initialConfig.k_error]
		if y_test[i] in top_k_classes:
			count += 1
	
	top_k_error = float(count) / float(X_test.shape[0])
	
	return top_k_error, baseline_accuracy, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro')

def fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test):
	
	"""
	This function is responsible for fitting a classifier
	to a training set and returning the classifier object
	for later use.
	
	Arguments
	---------
	X_train: :obj:`numpy array`
		array containing the features of the train set
	y_train: :obj:`numpy array`
		array containing the labels of the train set
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes 
	
	Returns
	-------
	clf: :obj:`scikit-learn classifier object`
		the trained classifier object
	"""
	
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
		
	elif clf_name == "Extra Trees":

		tuned_parameters = config.initialConfig.RandomForest_hyperparameters
		clf = ExtraTreesClassifier()
	
	elif clf_name == "MLP":
		
		tuned_parameters = config.initialConfig.MLP_hyperparameters
		clf = MLPClassifier()
	
	print(clf_name)
		
	for score in scores:

		clf = GridSearchCV(clf, tuned_parameters, cv=4,
						   scoring='%s' % score, verbose=0)
		clf.fit(X_train, y_train)
		
	return clf

def tuned_parameters_5_fold(poi_ids, conn, args):
	
	"""
	This function trains a collection of classifiers using
	a nested k-fold cross-validation approach and outputs
	the relevant results so that later comparisons can be made
	
	Arguments
	---------
	poi_ids: :obj:`list`
		the ids of the pois within the train set
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes 
	
	Returns
	-------
	None
	"""
	
	if config.initialConfig.experiment_folder == None:
		folderpath = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(folderpath)
		latest_folder = max(list_of_folders, key=os.path.getctime)
		args['folderpath'] = latest_folder
	
	# Shuffle ids	
	poi_ids = poi_ids[config.initialConfig.poi_id]
	random.shuffle(poi_ids)
		
	poi_ids = list(poi_ids)
			
	clf_names_not_tuned = ["Naive Bayes", "Gaussian Process", "QDA", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
	baseline_scores = []
	top_k_errors = []
	for item in clf_scores_dict:
		clf_scores_dict[item] = [[], [], []]
	report_data = []
	hyperparams_data = []

	kf = KFold(n_splits = config.initialConfig.k_fold_parameter)
	
	count = 1
	args['fold_number'] = count
	for train_ids, test_ids in kf.split(poi_ids):
		break
	train_ids = [train_id + 1 for train_id in train_ids]
	test_ids = [test_id + 1 for test_id in test_ids]
					
	# get train and test sets
	X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, train_ids, test_ids, count)
		
	most_common_classes = find_10_most_common_classes_train(y_train)
	
	# read clf name from csv
	row = {}
	
	if args['best_clf'] in clf_names_not_tuned:
		if clf_name == "Naive Bayes":
			clf = GaussianNB()
			clf.fit(X_train, y_train)
		elif clf_name == "Gaussian Process":
			clf = GaussianProcessClassifier()
			clf.fit(X_train, y_train)
		else:
			clf = AdaBoostClassifier()
			clf.fit(X_train, y_train)
	else:
		clf = fine_tune_parameters_given_clf(args['best_clf'], X_train, y_train, X_test, y_test)
	
	hyperparams_data = clf.best_params_
	df2 = pd.DataFrame.from_dict([hyperparams_data])
	
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'best_hyperparameters_' + str(args['level']) + '.csv'
			df2.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'best_hyperparameters_' + str(args['level']) + 'csv'
		df2.to_csv(filepath, index = False)
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=False,
		help="name of table containing pois information")
	ap.add_argument("-pois_csv_name", "--pois_csv_name", required=False,
		help="name of csv containing pois information")
	ap.add_argument("-best_hyperparameter_file_name", "--best_hyperparameter_file_name", required=False,
		help="desired name of best hyperparameter file")
	ap.add_argument("-best_clf_file_name", "--best_clf_file_name", required=False,
		help="name of file containing the best classifier found in step 1")

	args = vars(ap.parse_args())
	
	args['step'] = 2
		
	if args['pois_tbl_name'] is not None:
		print(args['pois_tbl_name'])
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	# get the poi ids
	if config.initialConfig.level == None:
		for level in [1,2]:
			if config.initialConfig.experiment_folder is not None:
				experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
				exists = os.path.isdir(experiment_folder_path)
				if exists:
					filepath = experiment_folder_path + '/' + 'best_clf_' + str(level) + '.csv'
					exists2 = os.path.isfile(filepath)
					if exists2:
						with open(filepath, 'r') as csv_file:
							reader = csv.reader(csv_file)
							count = 0
							for row in reader:
								if count == 1:
									args['best_clf'] = row[0]
								count += 1
					else:
						print("ERROR! No best_clf file found inside the folder")
						return
				else:
					print("ERROR! No experiment folder with the given name found")
					return
			else:
				experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
				list_of_folders = glob.glob(experiment_folder_path)
				if list_of_folders == []:
					print("ERROR! No experiment folder found inside the root folder")
					return
				else:
					latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
					filepath = latest_experiment_folder + '/' + 'best_clf_' + str(level) + '.csv'
					exists = os.path.isfile(filepath)
					if exists:
						with open(filepath, 'r') as csv_file:
							reader = csv.reader(csv_file)
							count = 0
							for row in reader:
								if count == 1:
									args['best_clf'] = row[0]
								count += 1
					else:
						print("ERROR! No best_clf file found inside the folder!")	
						return
			args['level'] = level
			poi_ids = get_poi_ids(conn, args)
			tuned_parameters_5_fold(poi_ids, conn, args)
	else:
		for level in config.initialConfig.level:
			if config.initialConfig.experiment_folder is not None:
				experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
				exists = os.path.isdir(experiment_folder_path)
				if exists:
					filepath = experiment_folder_path + '/' + 'best_clf_' + str(level) + '.csv'
					exists2 = os.path.isfile(filepath)
					if exists2:
						with open(filepath, 'r') as csv_file:
							reader = csv.reader(csv_file)
							count = 0
							for row in reader:
								if count == 1:
									args['best_clf'] = row[0]
								count += 1
					else:
						print("ERROR! No best_clf file found inside the folder")
						return
				else:
					print("ERROR! No experiment folder with the given name found")
					return
			else:
				experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
				list_of_folders = glob.glob(experiment_folder_path)
				if list_of_folders == []:
					print("ERROR! No experiment folder found inside the root folder")
					return
				else:
					latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
					filepath = latest_experiment_folder + '/' + 'best_clf_' + str(level) + '.csv'
					exists = os.path.isfile(filepath)
					if exists:
						with open(filepath, 'r') as csv_file:
							reader = csv.reader(csv_file)
							count = 0
							for row in reader:
								if count == 1:
									args['best_clf'] = row[0]
								count += 1
					else:
						print("ERROR! No best_clf file found inside the folder!")	
						return
			args['level'] = level
			poi_ids = get_poi_ids(conn, args)
			tuned_parameters_5_fold(poi_ids, conn, args)
	
if __name__ == "__main__":
   main()
