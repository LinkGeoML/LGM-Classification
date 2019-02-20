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
	k_preds = {}
	
	predictions = {}
	for k in config.initialConfig.k_error:
		predictions[k] = [[[] for _ in range(0, k)] for _ in range(0, X_test.shape[0])]
	
	for k in config.initialConfig.k_error:
		for i in range(0, X_test.shape[0]):
			top_k_classes = best_k_probs[:k]
			for j in range(0, k):
				predictions[i][j] = top_k_classes[j]
			if y_test[i] in top_k_classes:
				count += 1
		
		top_k_error = float(count) / float(X_test.shape[0])
		top_k_errors.append(top_k_error)
	#print("top_k_error: {0}".format(top_k_error))
	
	return top_k_errors, baseline_accuracy, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='weighted'), f1_score(y_test, y_pred, average='macro'), predictions
		
def tuned_parameters_5_fold(poi_ids, conn, args):
	
	# Shuffle ids
	poi_ids = poi_ids[config.initialConfig.poi_id]
	random.shuffle(poi_ids)
	poi_ids = list(poi_ids)
		
	X_test, y_test = get_train_set(conn, args, poi_ids)
		
	most_common_classes = find_10_most_common_classes_train(y_test)
	
	# Load model
	if args['trained_model_file_name'] is not None:
		model = joblib.load(args['trained_model_file_name'])
	else:
		list_of_files = glob.glob('trained_model*')
		latest_file = max(list_of_files, key=os.path.getctime)
		model = joblib.load(latest_file)
		
	top_k_error_list, baseline_accuracy, accuracy, f1_score_micro, f1_score_macro, predictions = get_score_for_10_most_common_classes(X_test, y_test, most_common_classes, model)	
	
	row = {}
	
	if args['results_file_name'] is not None:
		output_file = args['results_file_name']
	else:
		output_file = 'pred_categories'
	
	for k in config.initialConfig.k_error:
		count = 0
		for id, preds in zip(poi_ids, predictions[k]):
			for i in range(len(preds)):
				row['id'] = id
				row['pred{0}'.format(i)] = preds[i]
			out_df = pd.DataFrame([row])
			filename = output_file + '_' + str(k) + '_' + str(datetime.datetime.now()) + '.csv'
			if count == 1:
				with open(filename, 'a') as f:
					out_df.to_csv(f, index = False, header = True)
			else:
				with open(filename, 'a') as f:
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
