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
import glob
from sklearn.externals import joblib

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

from sklearn.preprocessing import LabelEncoder

import datetime

import config

np.random.seed(1234)

def get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, clf):
	
	top_class_count = 0
	for label in y_test:
		if label == most_common_classes[0]:
			top_class_count += 1
	
	baseline_accuracy = float(top_class_count) / float(y_test.shape[0])
	print(X_test.shape)
	y_pred = clf.predict(X_test)	
	
	probs = clf.predict_proba(X_test)
	sorted_prob_scores = np.sort(probs, axis = 1)
	best_k_probs = np.argsort(probs, axis = 1)
	count = 0
	top_k_errors = []
	k_preds = {}
	
	predictions = {}
	for k in config.initialConfig.k_error:
		predictions[k] = [[[] for _ in range(0, k)] for _ in range(0, X_test.shape[0])]
	
	prediction_scores = {}
	for k in config.initialConfig.k_error:
		prediction_scores[k] = [[[] for _ in range(0, k)] for _ in range(0, X_test.shape[0])]
	
	count = 0
	top_k_errors = []
	for k in config.initialConfig.k_error:
		for i in range(0, X_test.shape[0]):
			top_k_classes = best_k_probs[i][-k:]
			top_k_prediction_scores = sorted_prob_scores[i][-k:]
			for j in range(0, len(predictions[k][i])):
				predictions[k][i][j] = top_k_classes[j]
				prediction_scores[k][i][j] = top_k_prediction_scores[j]
			if y_test[i] in top_k_classes:
				count += 1
		
		top_k_error = float(count) / float(X_test.shape[0])
		top_k_errors.append(top_k_error)
		count = 0
	
	return top_k_errors, baseline_accuracy, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro'), predictions, prediction_scores
		
def tuned_parameters_5_fold(poi_ids, conn, args):
	
	if config.initialConfig.experiment_folder == None:
		folderpath = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(folderpath)
		latest_folder = max(list_of_folders, key=os.path.getctime)
		args['folderpath'] = latest_folder
	
	# Shuffle ids
	poi_ids = poi_ids[config.initialConfig.poi_id]
	random.shuffle(poi_ids)
	poi_ids = list(poi_ids)
		
	X_test, y_test = get_test_set(conn, args, poi_ids)
		
	most_common_classes = find_10_most_common_classes_train(y_test)
	
	# Load model
	if args['trained_model_file_name'] is not None:
		model = joblib.load(args['trained_model_file_name'])
	else:
		if config.initialConfig.experiment_folder == None:
			experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
			list_of_folders = glob.glob(experiment_folder_path)
			if list_of_folders == []:
				print("ERROR! No experiment folder found inside the root folder")
				return
			else:
				list_of_files = glob.glob(experiment_folder_path + '/trained_model*')
				latest_file = max(list_of_files, key=os.path.getctime)
				model = joblib.load(latest_file)
		else:
			experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
			list_of_files = glob.glob(experiment_folder_path + '/trained_model*')
			latest_file = max(list_of_files, key=os.path.getctime)
			model = joblib.load(latest_file)
		
		
	top_k_error_list, baseline_accuracy, accuracy, f1_score_micro, f1_score_macro, predictions, prediction_scores = get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, model)	
	#return
	row = {}
	
	if args['results_file_name'] is not None:
		output_file = args['results_file_name']
	else:
		output_file = 'pred_categories'
	
	for k in config.initialConfig.k_error:
		count = 0
		curr_time = str(datetime.datetime.now())
		curr_time = curr_time.replace(':', '.')
		for id, preds, scores in zip(poi_ids, predictions[k], prediction_scores[k]):
			for i in range(len(preds)):
				row['id'] = id

				row['prob_score{0}'.format(i)] = scores[i]
				row['pred{0}'.format(i)] = args['label_encoder'].inverse_transform(preds[i])
			out_df = pd.DataFrame([row])
			
			if config.initialConfig.experiment_folder == None:
				experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
				list_of_folders = glob.glob(experiment_folder_path)
				if list_of_folders == []:
					print("ERROR! No experiment folder found inside the root folder")
					return
				else:
					latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
					filepath = latest_experiment_folder + '/' + output_file + '_' + str(args['level']) + '_top' + str(k) + 'categories.csv'
					if count == 0:
						with open(filepath, 'a') as f:
							out_df.to_csv(f, index = False, header = True)
					else:
						with open(filepath, 'a') as f:
							out_df.to_csv(f, index = False, header = False)
					count += 1
			else:
				experiment_folder_path = config.initialConfig.root_path + config.initialConfig.experiment_folder
				filepath = experiment_folder_path + '/' + output_file + '_' + str(args['level']) + '_top' + str(k) + 'categories.csv'
				if count == 0:
					with open(filepath, 'a') as f:
						out_df.to_csv(f, index = False, header = True)
				else:
					with open(filepath, 'a') as f:
						out_df.to_csv(f, index = False, header = False)
				count += 1
	
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
	ap.add_argument("-trained_model_file_name", "--trained_model_file_name", required=False,
		help="name of pickle file containing the model")
	ap.add_argument("-results_file_name", "--results_file_name", required=False,
		help="desired name of output file")

	args = vars(ap.parse_args())
	
	if args['pois_tbl_name'] is not None:
		print(args['pois_tbl_name'])
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	args['step'] = 4
	
	print(args['pois_csv_name'])
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
