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

import datetime
import config
import glob
import os
import csv

np.random.seed(1234)
		
def tuned_parameters_5_fold(poi_ids, conn, args):
	
	if config.initialConfig.experiment_folder == None:
		folderpath = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(folderpath)
		latest_folder = max(list_of_folders, key=os.path.getctime)
		args['folderpath'] = latest_folder
	
	from sklearn.externals import joblib
	
	# Shuffle ids
	poi_ids = poi_ids[config.initialConfig.poi_id]
	random.shuffle(poi_ids)	
	poi_ids = list(poi_ids)
			
	clf_names_not_tuned = ["Naive Bayes", "MLP", "Gaussian Process", "QDA", "AdaBoost"]
	clf_names = config.initialConfig.classifiers
	clf_scores_dict = dict.fromkeys(clf_names)
		
	# get train and test sets
	X_train, y_train = get_train_set(conn, args, poi_ids)
		
	most_common_classes = find_10_most_common_classes_train(y_train)
	
	# read clf name from csv
	row = {}
	
	if args['best_clf'] in clf_names_not_tuned:
		if clf_name == "Naive Bayes":
			clf = GaussianNB()
			clf.fit(X_train, y_train)
		#elif clf_name == "MLP":
		#	clf = MLPClassifier()
		#	clf.fit(X_train, y_train)
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
		# read hyperparameters and fit the model
		clf = train_clf_given_hyperparams(X_train, y_train, args)
		
	# store the model as a pickle file
	
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			filepath = latest_experiment_folder + '/' + 'trained_model_' + str(args['level']) + '.pkl'
			joblib.dump(clf, filepath, compress = 9)
	else:
		experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
		filepath = experiment_folder_path + '/' + 'trained_model_' + str(args['level']) + '.pkl'
		joblib.dump(clf, filepath, compress = 9)

def train_clf_given_hyperparams(X_train, y_train, args):
	tuned_parameters = args['best_hyperparams']
	for parameter in tuned_parameters:
		#print(parameter)
		if tuned_parameters[parameter].isdigit():
			#print("edw")
			tuned_parameters[parameter] = float(tuned_parameters[parameter])
	#tuned_parameters['degree'] = int(tuned_parameters['degree'])
	print(tuned_parameters)
	
	if args['best_clf'] == "SVM":
		clf = SVC(probability=True, **tuned_parameters)
		
	elif args['best_clf'] == "Nearest Neighbors":
		clf = KNeighborsClassifier(**tuned_parameters)
		
	elif args['best_clf'] == "Decision Tree":
		clf = DecisionTreeClassifier(**tuned_parameters)
		
	elif args['best_clf'] == "Random Forest":
		clf = RandomForestClassifier(**tuned_parameters)
		
	elif args['best_clf'] == "Extra Trees":
		clf = ExtraTreesClassifier(**tuned_parameters)
		
	elif args['best_clf'] == "MLP":
		clf = MLPClassifier(**tuned_parameters)
	
	print(clf)
	print(X_train.shape)
	clf.fit(X_train, y_train)
		
	return clf

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
		help="desired name of best hyperparameter file")
	ap.add_argument("-trained_model_file_name", "--trained_model_file_name", required=False,
		help="name of file containing the trained model")

	args = vars(ap.parse_args())
		
	if args['pois_tbl_name'] is not None:
		print(args['pois_tbl_name'])
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	args['step'] = 3
	
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
						#input_file = csv.DictReader(open(filepath))
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
					print("ERROR! Given best_clf file does not exist!")
					return
			else:
				experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
				list_of_folders = glob.glob(experiment_folder_path)
				if list_of_folders == []:
					print("ERROR! No experiment folder found")
					return
				else:
					latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
					filepath = latest_experiment_folder + '/' + 'best_clf_' + str(level) + '.csv'
					exists = os.path.isfile(filepath)
					if exists:
						#input_file = csv.DictReader(open(filepath))
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
					
			#print(args['best_clf'])
			
			if config.initialConfig.experiment_folder is not None:
				experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
				exists = os.path.isdir(experiment_folder_path)
				if exists:
					filepath = experiment_folder_path + '/' + 'best_hyperparameters_' + str(level) + '.csv'
					exists2 = os.path.isfile(filepath)
					if exists2:
						input_file = csv.DictReader(open(filepath))
						for row in input_file:
							hyperparams_dict = row
					else:
						print("ERROR! No best_hyperparameters file found inside the folder")
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
					filepath = latest_experiment_folder + '/' + 'best_hyperparameters_' + str(level) + '.csv'
					exists = os.path.isfile(filepath)
					if exists:
						input_file = csv.DictReader(open(filepath))
						for row in input_file:
							hyperparams_dict = row
					else:
						print("ERROR! No best_hyperparameters file found inside the folder!")	
						return		
						
			args['best_hyperparams'] = hyperparams_dict
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
						#input_file = csv.DictReader(open(filepath))
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
					print("ERROR! Given best_clf file does not exist!")
					return
			else:
				experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
				list_of_folders = glob.glob(experiment_folder_path)
				if list_of_folders == []:
					print("ERROR! No experiment folder found")
					return
				else:
					latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
					filepath = latest_experiment_folder + '/' + 'best_clf_' + str(level) + '.csv'
					exists = os.path.isfile(filepath)
					if exists:
						#input_file = csv.DictReader(open(filepath))
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
					
			#print(args['best_clf'])
			
			if config.initialConfig.experiment_folder is not None:
				experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
				exists = os.path.isdir(experiment_folder_path)
				if exists:
					filepath = experiment_folder_path + '/' + 'best_hyperparameters_' + str(level) + '.csv'
					exists2 = os.path.isfile(filepath)
					if exists2:
						input_file = csv.DictReader(open(filepath))
						for row in input_file:
							hyperparams_dict = row
					else:
						print("ERROR! No best_hyperparameters file found inside the folder")
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
					filepath = latest_experiment_folder + '/' + 'best_hyperparameters_' + str(level) + '.csv'
					exists = os.path.isfile(filepath)
					if exists:
						input_file = csv.DictReader(open(filepath))
						for row in input_file:
							hyperparams_dict = row
					else:
						print("ERROR! No best_hyperparameters file found inside the folder!")	
						return		
						
			args['best_hyperparams'] = hyperparams_dict
			args['level'] = level
			poi_ids = get_poi_ids(conn, args)
			tuned_parameters_5_fold(poi_ids, conn, args)
	
if __name__ == "__main__":
   main()
