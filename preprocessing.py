#!/usr/bin/python

import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from pois_feature_extraction import *
from textual_feature_extraction import *
from geospatial_feature_extraction import *
from pois_feature_extraction_csv import *
from feml import *
import nltk
import config

def get_train_test_poi_ids(conn, args):
	
	from sklearn.model_selection import train_test_split
	
	if args["pois_tbl_name"] is not None:
		# get all poi details
		sql = "select {0}.id as poi_id, {0}.geom from {0}".format(args["pois_tbl_name"])
		df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
		X = np.asarray(df[config.initialConfig.poi_id])
		y = np.zeros(len(df[config.initialConfig.poi_id]))
	else:
		df = pd.read_csv(args['pois_csv_name'])
		X = np.asarray(df[config.initialConfig.poi_id])
		y = np.zeros(len(df[config.initialConfig.poi_id]))
	
	poi_ids_train, poi_ids_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        
	return poi_ids_train, poi_ids_test
	
def get_poi_ids(conn, args):
	
	from sklearn.model_selection import train_test_split
	
	if args["pois_tbl_name"] is not None:
		# get all poi details
		sql = "select {0}.id as poi_id, {0}.geom from {0}".format(args["pois_tbl_name"])
		df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	else:
		df = pd.read_csv(args['pois_csv_name'])
	
	"""
	# get all poi details
	sql = "select {0}.id as poi_id, {0}.geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	"""
	#ids = np.asarray(df['poi_id'])
        
	return df
        
def get_train_test_sets(conn, args, poi_ids_train, poi_ids_test):
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	if args["pois_tbl_name"] is not None:
		poi_df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict(conn, args)
		
		# we read the different labels
		class_codes_set = get_class_codes_set(args, poi_df)
	else:
		poi_df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict_csv(conn, args)
		
		# we read the different labels
		class_codes_set = get_class_codes_set_csv(args, poi_df)
		
	# we encode them so we can have a more compact representation of them
	poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict(class_codes_set, poi_id_to_class_code_coordinates_dict)
	#print(encoded_labels_set)
	
	y_train = []
	y_test = []
	
	X_train = []
	X_test = []
	
	#print(poi_ids_train)
	#print(poi_ids_test)
		
	for poi_id in poi_ids_train:
		#print(poi_id)
		#print(poi_id_to_encoded_labels_dict[poi_id][0][0])
		y_train.append(poi_id_to_encoded_labels_dict[poi_id][0][0])
	for poi_id in poi_ids_test:
		#print(poi_id)
		#print(poi_id_to_encoded_labels_dict[poi_id][0][0])
		y_test.append(poi_id_to_encoded_labels_dict[poi_id][0][0])
		
	y_train = np.asarray(y_train)
	y_test = np.asarray(y_test)
	
	poi_id_to_class_centroid_similarities_train, encoded_labels_corpus_train = get_poi_id_to_class_centroid_similarities(poi_ids_train, poi_id_to_class_code_coordinates_dict, encoded_labels_set, conn, args, [])
	poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids_train, conn, args, config.initialConfig.top_k_character_ngrams_percentage)
	poi_id_to_word_features = get_features_top_k(poi_ids_train, conn, args, config.initialConfig.top_k_terms_percentage)
	poi_id_to_word_features_ngrams_tokens = get_features_top_k_ngrams_tokens(poi_ids_train, conn, args, config.initialConfig.top_k_terms_percentage)
	
	if args["pois_tbl_name"] is not None: 
		closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids_train, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
		closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids_train, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
	else:
		closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label_csv(poi_ids_train, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
		closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets_csv(poi_ids_train, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
	
	count = 0
	for poi_id in poi_ids_train:
		#print(poi_id_to_class_centroid_similarities_train[poi_id])
		temp_feature_list1 = [item for sublist in closest_pois_boolean_and_counts_per_label[poi_id] for item in sublist]
		temp_feature_list2 = [item for sublist in closest_pois_boolean_and_counts_per_label_streets[poi_id] for item in sublist]
		
		if count == 0:
			print("poi id to nearby poi labels: {0}".format(len(temp_feature_list1)))
			print("poi id to nearby street labels: {0}".format(len(temp_feature_list2)))
			print("poi id to class centroid similarities: {0}".format(len(poi_id_to_class_centroid_similarities_train[poi_id])))
			print("poi id to roken features: {0}".format(len(poi_id_to_word_features[poi_id])))
			print("poi id to n-gram token features: {0}".format(len(poi_id_to_word_features_ngrams_tokens[poi_id])))
			print("poi id to n-gram features: {0}".format(len(poi_id_to_word_features_ngrams[poi_id])))
			
		count += 1
		
		feature_list = poi_id_to_class_centroid_similarities_train[poi_id] + poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + poi_id_to_word_features_ngrams_tokens[poi_id] + temp_feature_list1 + temp_feature_list2
		X_train.append(feature_list)
	
	poi_id_to_class_centroid_similarities_test, encoded_labels_corpus_test = get_poi_id_to_class_centroid_similarities(poi_ids_test, poi_id_to_class_code_coordinates_dict, encoded_labels_set, conn, args, encoded_labels_corpus_train, test = True)
	poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids_train, conn, args, config.initialConfig.top_k_character_ngrams_percentage, poi_ids_test)
	poi_id_to_word_features = get_features_top_k(poi_ids_train, conn, args, config.initialConfig.top_k_terms_percentage, poi_ids_test)
	poi_id_to_word_features_ngrams_tokens = get_features_top_k_ngrams_tokens(poi_ids_train, conn, args, config.initialConfig.top_k_terms_percentage, poi_ids_test)
	
	if args["pois_tbl_name"] is not None:
		closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids_test, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
		closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids_test, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
	else:
		closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label_csv(poi_ids_test, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
		closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets_csv(poi_ids_test, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
	
	for poi_id in poi_ids_test:
		temp_feature_list1 = [item for sublist in closest_pois_boolean_and_counts_per_label[poi_id] for item in sublist]
		temp_feature_list2 = [item for sublist in closest_pois_boolean_and_counts_per_label_streets[poi_id] for item in sublist]
		feature_list = poi_id_to_class_centroid_similarities_test[poi_id] + poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + poi_id_to_word_features_ngrams_tokens[poi_id] + temp_feature_list1 + temp_feature_list2
		X_test.append(feature_list)

	X_train = np.asarray(X_train)
	X_test = np.asarray(X_test)
	
	print(X_train.shape)
	print(X_test.shape)
		
	return X_train, y_train, X_test, y_test
	
def get_train_set(conn, args, poi_ids):
	
	#print(len(poi_ids))
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	if args["pois_tbl_name"] is not None:
		poi_df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict(conn, args)
		
		# we read the different labels
		class_codes_set = get_class_codes_set(args, poi_df)
	else:
		poi_df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict_csv(conn, args)
		
		# we read the different labels
		class_codes_set = get_class_codes_set_csv(args, poi_df)
		
	# we encode them so we can have a more compact representation of them
	poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict(class_codes_set, poi_id_to_class_code_coordinates_dict)
	#print(encoded_labels_set)
	#print(poi_id_to_encoded_labels_dict)
	y_train = []
	
	X_train = []
	
	#print(poi_ids_train)
	#print(poi_ids_test)
		
	for poi_id in poi_ids:
		#print(poi_id)
		#print(poi_id_to_encoded_labels_dict[poi_id][0][0])
		y_train.append(poi_id_to_encoded_labels_dict[poi_id][0][0])
		
	y_train = np.asarray(y_train)
	#print(y_train)
	
	#print(y_train.shape[0])
	#print(len(poi_ids))
	poi_id_to_class_centroid_similarities_train, encoded_labels_corpus_train = get_poi_id_to_class_centroid_similarities(poi_ids, poi_id_to_class_code_coordinates_dict, encoded_labels_set, conn, args, [])
	poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids, conn, args, config.initialConfig.top_k_character_ngrams_percentage)
	poi_id_to_word_features = get_features_top_k(poi_ids, conn, args, config.initialConfig.top_k_terms_percentage)
	poi_id_to_word_features_ngrams_tokens = get_features_top_k_ngrams_tokens(poi_ids, conn, args, config.initialConfig.top_k_terms_percentage)
	
	#print(poi_id_to_class_centroid_similarities_train, poi_id_to_word_features_ngrams, poi_id_to_word_features, poi_id_to_word_features_ngrams_tokens)
	
	if args["pois_tbl_name"] is not None: 
		closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
		closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
	else:
		closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label_csv(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
		closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets_csv(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
	
	#print(closest_pois_boolean_and_counts_per_label_streets)
	count = 0
	for poi_id in poi_ids:
		#print(poi_id_to_class_centroid_similarities_train[poi_id])
		temp_feature_list1 = [item for sublist in closest_pois_boolean_and_counts_per_label[poi_id] for item in sublist]
		temp_feature_list2 = [item for sublist in closest_pois_boolean_and_counts_per_label_streets[poi_id] for item in sublist]
		
		if count == 0:
			print("poi id to nearby poi labels: {0}".format(len(temp_feature_list1)))
			print("poi id to nearby street labels: {0}".format(len(temp_feature_list2)))
			print("poi id to class centroid similarities: {0}".format(len(poi_id_to_class_centroid_similarities_train[poi_id])))
			print("poi id to roken features: {0}".format(len(poi_id_to_word_features[poi_id])))
			print("poi id to n-gram token features: {0}".format(len(poi_id_to_word_features_ngrams_tokens[poi_id])))
			print("poi id to n-gram features: {0}".format(len(poi_id_to_word_features_ngrams[poi_id])))
			
		count += 1
		
		feature_list = poi_id_to_class_centroid_similarities_train[poi_id] + poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + poi_id_to_word_features_ngrams_tokens[poi_id] + temp_feature_list1 + temp_feature_list2
		X_train.append(feature_list)
	
	X_train = np.asarray(X_train)
	print(X_train.shape)
		
	return X_train, y_train
	

def standardize_data(X_train, X_test):
	from sklearn.preprocessing import StandardScaler
	
	standard_scaler = StandardScaler()
	X_train = standard_scaler.fit_transform(X_train)
	X_test = standard_scaler.transform(X_test)
	
	return X_train, X_test
	
def find_10_most_common_classes_train(y_train):
	
	labels = list(y_train)
	
	label_counter = {}
	for label in labels:
		if label in label_counter:
			label_counter[label] += 1
		else:
			label_counter[label] = 1

	classes_ranked = sorted(label_counter, key = label_counter.get, reverse = True) 
	most_common_classes = classes_ranked[:10]
	
	return most_common_classes

