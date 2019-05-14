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

from numpy import genfromtxt

import datetime

import config

np.random.seed(1234)

def get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
	top_class_count = 0
	for label in y_test:
		if label == most_common_classes[0]:
			top_class_count += 1
	
	baseline_accuracy = float(top_class_count) / float(y_test.shape[0])
	y_pred = clf.predict(X_test)	
	
	baseline_preds = np.ones(X_test.shape[0], dtype=int)
	baseline_preds = baseline_preds * most_common_classes[0]
	baseline_f_score = f1_score(y_test, baseline_preds, average='weighted')
	
	probs = clf.predict_proba(X_test)
	count = 0
	top_k_errors = []
	for k in config.initialConfig.k_error:
		best_k_probs = np.argsort(probs, axis = 1)[:,-k:]
		for i in range(0, X_test.shape[0]):
			top_k_classes = best_k_probs[i]
			top_k_classes[-1] = y_pred[i]
			if y_test[i] in top_k_classes:
				count += 1
		
		top_k_error = float(count) / float(X_test.shape[0])
		top_k_errors.append(top_k_error)
		count = 0
	
	return top_k_errors, baseline_accuracy, baseline_f_score, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro')

def fine_tune_parameters_given_clf(clf_name, X_train, y_train, X_test, y_test):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
	
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
	
def feature_selection(X_train, X_test, y_train):
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""
		
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
	feature_mask = sel.get_support()
	X_test_new = np.zeros((X_test.shape[0], X_train.shape[1]))
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
	
	"""
	This function is responsible for mapping the pois to a list of two-element lists.
	The first element of that list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second
	element contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	
	Arguments
	---------
	num_of_labels: the total number of the different labels
	encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold
	
	Returns
	-------
	"""

	if config.initialConfig.experiment_folder == None:
		folderpath = config.initialConfig.root_path + 'experiment_folder_' + str(datetime.datetime.now())
		folderpath = folderpath.replace(':', '-')
		os.makedirs(folderpath)
		args['folderpath'] = folderpath
	
	# Shuffle ids
	poi_ids = poi_ids[config.initialConfig.poi_id]
	random.shuffle(poi_ids)
	
	kf = KFold(n_splits = config.initialConfig.k_fold_parameter)
	
	count = 1
	
	clf_names_not_tuned = ["Naive Bayes", "Gaussian Process", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
	baseline_scores = []
	baseline_scores2 = []
	top_k_errors = [[] for _ in range(0, len(config.initialConfig.k_error))]
	for item in clf_scores_dict:
		clf_scores_dict[item] = [[], [], []]
	
	clf_k_error_scores_dict = dict.fromkeys(clf_names)
	for item in clf_k_error_scores_dict:
		clf_k_error_scores_dict[item] = [[] for _ in range(len(config.initialConfig.k_error))] 
	
	report_data = []
	hyperparams_data = []
	
	# split data into train, test
	for train_ids, test_ids in kf.split(poi_ids):
		
		args['fold_number'] = count
		
		train_ids = [train_id + 1 for train_id in train_ids]
		test_ids = [test_id + 1 for test_id in test_ids]

		#X_train = genfromtxt('X_train_fold{0}'.format(count), delimiter = ',')
		#y_train = genfromtxt('y_train_fold{0}'.format(count), delimiter = ',')
		#X_test = genfromtxt('X_test_fold{0}'.format(count), delimiter = ',')
		#y_test = genfromtxt('y_test_fold{0}'.format(count), delimiter = ',')
			
		# get train and test sets
		X_train, y_train, X_test, y_test = get_train_test_sets(conn, args, train_ids, test_ids, count)
		
		#np.savetxt("X_train_fold{0}.csv".format(count), X_train, delimiter=",")
		#np.savetxt("y_train_fold{0}.csv".format(count), y_train, delimiter=",")
		#np.savetxt("X_test_fold{0}.csv".format(count), X_test, delimiter=",")
		#np.savetxt("y_test_fold{0}.csv".format(count), y_test, delimiter=",")
		#count += 1
		#continue
		
		#X_train, X_test = feature_selection(X_train, X_test, y_train)
		
		most_common_classes = find_10_most_common_classes_train(y_train)
		
		clf = None
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
			
			top_k_error_list, baseline_accuracy, baseline_f_score, accuracy, f1_score_micro, f1_score_macro = get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf)
			clf_scores_dict[clf_name][0].append(accuracy)
			clf_scores_dict[clf_name][1].append(f1_score_micro)
			clf_scores_dict[clf_name][2].append(f1_score_macro)
			row['Fold'] = count
			row['Classifier'] = clf_name
			row['Baseline Accuracy'] = baseline_accuracy
			row['Baseline F-score'] = baseline_f_score
			i = 0
			for k, top_k_error in zip(config.initialConfig.k_error, top_k_error_list):
				row['Top-{0} Accuracy'.format(k)] = top_k_error
				top_k_errors[i].append(top_k_error)
				clf_k_error_scores_dict[clf_name][i].append(top_k_error)
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
			baseline_scores2.append(baseline_f_score)
			
		count += 1
		
	for clf_name in clf_names:	
		row = {}
		row['Fold'] = 'average'
		row['Classifier'] = clf_name
		row['Accuracy'] = sum(map(float,clf_scores_dict[clf_name][0])) / 5.0 
		i = 0
		for k in config.initialConfig.k_error:
			row['Top-{0} Accuracy'.format(k)] = float(sum(map(float, clf_k_error_scores_dict[clf_name][i]))) / 5.0
			i += 1
		row['Baseline Accuracy'] = sum(map(float,baseline_scores)) / 40.0
		row['Baseline F-score'] = sum(map(float,baseline_scores2)) / 40.0
		row['F1-Score-Micro'] = sum(map(float,clf_scores_dict[clf_name][1])) / 5.0
		row['F1-Score-Macro'] = sum(map(float,clf_scores_dict[clf_name][2])) / 5.0
		report_data.append(row)
		
	df = pd.DataFrame.from_dict(report_data)
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'classification_report_' + str(args['level']) + '.csv'
			df.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'classification_report_' + str(args['level']) + '.csv'
		df.to_csv(filepath, index = False)
	
	best_clf_row = {}
	best_clf_row['best_clf_score'] = 0.0
	best_clf_row['best_clf_name'] = ''
	for index, row in df.iterrows():
		if row['Fold'] == 'average':
			if row['Accuracy'] > best_clf_row['best_clf_score'] :
				best_clf_row['best_clf_score']  = row['Accuracy']
				best_clf_row['best_clf_name'] = row['Classifier']
	df2 = pd.DataFrame.from_dict([best_clf_row])
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'best_clf_' + str(args['level']) + '.csv'
			df2.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'best_clf_' + str(args['level']) + '.csv'
		df2.to_csv(filepath, index = False)
	
	df3 = pd.DataFrame.from_dict(hyperparams_data)
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'hyperparameters_per_fold_' + str(args['level']) + '.csv'
			df3.to_csv(filepath, index = False)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'hyperparameters_per_fold_' + str(args['level']) + '.csv'
		df3.to_csv(filepath, index = False)
	
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
	#ap.add_argument("-retrain_csv_file_name", "--retrain", required=False,
	#	help="name of csv containing the dataset you want to retrain the algorithm on")

	args = vars(ap.parse_args())
	
	if args['pois_tbl_name'] is not None:
		print(args['pois_tbl_name'])
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	args['step'] = 1
	
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
