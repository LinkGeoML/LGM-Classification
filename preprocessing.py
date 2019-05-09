#!/usr/bin/python

import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
from pois_feature_extraction_csv import *
from textual_feature_extraction import *
from geospatial_feature_extraction import *
from feml import *
import nltk
import config
import random

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
        
def get_train_test_sets(conn, args, poi_ids_train, poi_ids_test, fold_number = None):
	
	args['fold_number'] = fold_number
	
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
	label_encoder, poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict_csv(args, class_codes_set, poi_id_to_class_code_coordinates_dict)
	args['label_encoder'] = label_encoder
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
	
	feature_dict = dict((el, None) for el in config.initialConfig.feature_list)
	for key in config.initialConfig.included_features:
		feature_dict[key] = []
	
	sentinel = 0
	for key in config.initialConfig.included_features:
		if fold_number is not None:
			if config.initialConfig.experiment_folder == None:
				filepath = args['folderpath'] + '/' + key + '_' + 'train_' + str(fold_number) + '.csv'
			else:
				filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + key + '_' + 'train_' + str(fold_number) + '.csv'
		else:
			if config.initialConfig.experiment_folder == None:
				filepath = args['folderpath'] + '/' + key + '_' + 'train' + '.csv'
			else:	
				filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + key + '_' + 'train' + '.csv'
		#print(filepath)
		exists = os.path.isfile(filepath)
		if exists:
			if sentinel == 0:
				X_train = np.genfromtxt(filepath, delimiter=',')
				sentinel = 1
			else:
				temp_array = np.genfromtxt(filepath, delimiter=',')
				X_train = np.concatenate((X_train, temp_array), axis = 1)
				#sentinel = 1
			
	"""	
	sentinel = 0
	for key in config.initialConfig.included_features:
		if fold_number is not None:
			filepath = config.initialConfig.root_path + key + '_' + 'train_' + str(fold_number) + '.csv'
		else:
			filepath = config.initialConfig.root_path + key + '_' + 'train' + '.csv'
		exists = os.path.isdir(filepath)
		if exists:
			if sentinel == 0:
				X_train = np.genfromtxt(filepath, delimiter=',')
				sentinel = 1
			else:
				temp_array = np.genfromtxt(filepath, delimiter=',')
				X_train = np.concatenate((X_train, temp_array), axis = 1)
				#sentinel = 1
	"""
				
	if not exists:
		scaler_dict = dict((el, None) for el in config.initialConfig.included_features)
		
		if feature_dict['class_centroid_similarities'] is not None:
			poi_id_to_class_centroid_similarities_train, encoded_labels_corpus_train = get_poi_id_to_class_centroid_similarities(poi_ids_train, poi_id_to_class_code_coordinates_dict, encoded_labels_set, conn, args, [])
		if feature_dict['word_features_ngrams'] is not None:
			poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids_train, conn, args, config.initialConfig.top_k_character_ngrams_percentage)
		if feature_dict['word_features'] is not None:
			poi_id_to_word_features = get_features_top_k(poi_ids_train, conn, args, config.initialConfig.top_k_terms_percentage)
		if feature_dict['word_features_ngrams_tokens'] is not None:
			poi_id_to_word_features_ngrams_tokens = get_features_top_k_ngrams_tokens(poi_ids_train, conn, args, config.initialConfig.top_k_terms_percentage)
		
		if args["pois_tbl_name"] is not None:
			if feature_dict['poi_to_poi_radius_boolean'] is not None or feature_dict['poi_to_poi_radius_counts'] is not None: 
				closest_pois_boolean_per_label, closest_pois_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids_train, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None or feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 	
				closest_pois_boolean_per_label_streets, closest_pois_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids_train, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
		else:
			if feature_dict['poi_to_poi_radius_boolean'] is not None or feature_dict['poi_to_poi_radius_counts'] is not None: 
				closest_pois_boolean_per_label, closest_pois_counts_per_label = get_closest_pois_boolean_and_counts_per_label_csv(poi_ids_train, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None or feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 	
				closest_pois_boolean_per_label_streets, closest_pois_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets_csv(poi_ids_train, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
		
		if feature_dict['poi_to_poi_neighbors_boolean'] is not None or feature_dict['poi_to_poi_neighbors_counts'] is not None:
			poi_to_poi_neighbors_boolean, poi_to_poi_neighbors_counts = get_neighbor_pois_boolean_and_counts_per_label_csv(poi_ids_train, conn, args)	
		
		#print(poi_id_to_word_features)
		#count = 0
		for poi_id in poi_ids_train:
			if feature_dict['poi_to_poi_radius_boolean'] is not None:
				temp_feature_list1 = closest_pois_boolean_per_label[poi_id]
			if feature_dict['poi_to_poi_radius_counts'] is not None: 
				temp_feature_list2 = closest_pois_counts_per_label[poi_id]
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None:
				temp_feature_list3 = closest_pois_boolean_per_label_streets[poi_id]
			if feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 
				temp_feature_list4 = closest_pois_counts_per_label_streets[poi_id]
			if feature_dict['poi_to_poi_neighbors_boolean'] is not None:
				temp_feature_list5 = poi_to_poi_neighbors_boolean[poi_id]
			if feature_dict['poi_to_poi_neighbors_counts'] is not None:
				temp_feature_list6 = poi_to_poi_neighbors_counts[poi_id]

			"""
			if count == 0:
				print("poi id to nearby poi labels: {0}".format(len(temp_feature_list1)))
				print("poi id to nearby street labels: {0}".format(len(temp_feature_list2)))
				print("poi id to class centroid similarities: {0}".format(len(poi_id_to_class_centroid_similarities_train[poi_id])))
				print("poi id to roken features: {0}".format(len(poi_id_to_word_features[poi_id])))
				print("poi id to n-gram token features: {0}".format(len(poi_id_to_word_features_ngrams_tokens[poi_id])))
				print("poi id to n-gram features: {0}".format(len(poi_id_to_word_features_ngrams[poi_id])))
				
			count += 1
			"""
			if feature_dict['class_centroid_similarities'] is not None:
				feature_dict['class_centroid_similarities'].append(poi_id_to_class_centroid_similarities_train[poi_id])
			if feature_dict['word_features'] is not None:
				feature_dict['word_features'].append(poi_id_to_word_features[poi_id])
			if feature_dict['word_features_ngrams'] is not None:
				feature_dict['word_features_ngrams'].append(poi_id_to_word_features_ngrams[poi_id])
			if feature_dict['word_features_ngrams_tokens'] is not None:
				feature_dict['word_features_ngrams_tokens'].append(poi_id_to_word_features_ngrams_tokens[poi_id])
			if feature_dict['poi_to_poi_radius_boolean'] is not None:
				feature_dict['poi_to_poi_radius_boolean'].append(temp_feature_list1)
			if feature_dict['poi_to_poi_radius_counts'] is not None: 
				feature_dict['poi_to_poi_radius_counts'].append(temp_feature_list2)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None: 
				feature_dict['poi_to_closest_street_to_poi_radius_boolean'].append(temp_feature_list3)
			if feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 
				feature_dict['poi_to_closest_street_to_poi_radius_counts'].append(temp_feature_list4)
			if feature_dict['poi_to_poi_neighbors_boolean'] is not None: 
				feature_dict['poi_to_poi_neighbors_boolean'].append(temp_feature_list5)
			if feature_dict['poi_to_poi_neighbors_counts'] is not None: 
				feature_dict['poi_to_poi_neighbors_counts'].append(temp_feature_list6)
			
			#feature_list = poi_id_to_class_centroid_similarities_train[poi_id] + poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + poi_id_to_word_features_ngrams_tokens[poi_id] + temp_feature_list1 + temp_feature_list2
			#X_train.append(feature_list)
			
		if config.initialConfig.experiment_folder == None:
			folderpath = args['folderpath']
		else:
			folderpath = config.initialConfig.experiment_folder
			
		sentinel = 0
		for key in feature_dict:
			if feature_dict[key] is not None:
				if sentinel == 0:
					X_train = np.asarray(feature_dict[key])
					if key in config.initialConfig.features_to_normalize:
						X_train, scaler_dict[key] = standardize_data_train(X_train)
					if fold_number is not None:
						filepath = folderpath + '/' + key + '_' + 'train_' + str(fold_number) + '.csv'
						np.savetxt(filepath, X_train, delimiter=",")
					else:
						filepath = folderpath + '/' + key + '_' + 'train' + '.csv'
						np.savetxt(filepath, X_train, delimiter=",")
					sentinel = 1
				else:
					temp_array = np.asarray(feature_dict[key])
					if key in config.initialConfig.features_to_normalize:
						temp_array, scaler_dict[key] = standardize_data_train(temp_array)
					if fold_number is not None:
						filepath = folderpath + '/' + key + '_' + 'train_' + str(fold_number) + '.csv'
						np.savetxt(filepath, temp_array, delimiter=",")
					else:
						filepath = folderpath + '/' + key + '_' + 'train' + '.csv'
						np.savetxt(filepath, temp_array, delimiter=",")
					X_train = np.concatenate((X_train, temp_array), axis = 1)
					#sentinel = 1
						
		feature_dict = dict((el, None) for el in config.initialConfig.feature_list)
		for key in config.initialConfig.included_features:
			feature_dict[key] = []
		
		if feature_dict['class_centroid_similarities'] is not None:
			poi_id_to_class_centroid_similarities_test, encoded_labels_corpus_test = get_poi_id_to_class_centroid_similarities(poi_ids_test, poi_id_to_class_code_coordinates_dict, encoded_labels_set, conn, args, [])
		if feature_dict['word_features_ngrams'] is not None:
			poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids_test, conn, args, config.initialConfig.top_k_character_ngrams_percentage)
		if feature_dict['word_features'] is not None:
			poi_id_to_word_features = get_features_top_k(poi_ids_test, conn, args, config.initialConfig.top_k_terms_percentage)
		if feature_dict['word_features_ngrams_tokens'] is not None:
			poi_id_to_word_features_ngrams_tokens = get_features_top_k_ngrams_tokens(poi_ids_test, conn, args, config.initialConfig.top_k_terms_percentage)
		
		if args["pois_tbl_name"] is not None:
			if feature_dict['poi_to_poi_radius_boolean'] is not None or feature_dict['poi_to_poi_radius_counts'] is not None: 
				closest_pois_boolean_per_label, closest_pois_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids_test, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None or feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 		
				closest_pois_boolean_per_label_streets, closest_pois_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids_test, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
		else:
			if feature_dict['poi_to_poi_radius_boolean'] is not None or feature_dict['poi_to_poi_radius_counts'] is not None: 
				closest_pois_boolean_per_label, closest_pois_counts_per_label = get_closest_pois_boolean_and_counts_per_label_csv(poi_ids_test, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None or feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 	
				closest_pois_boolean_per_label_streets, closest_pois_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets_csv(poi_ids_test, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
		
		if feature_dict['poi_to_poi_neighbors_boolean'] is not None or feature_dict['poi_to_poi_neighbors_counts'] is not None:
			poi_to_poi_neighbors_boolean, poi_to_poi_neighbors_counts = get_neighbor_pois_boolean_and_counts_per_label_csv(poi_ids_test, conn, args)
		
		for poi_id in poi_ids_test:
			if feature_dict['poi_to_poi_radius_boolean'] is not None:
				temp_feature_list1 = closest_pois_boolean_per_label[poi_id]
			if feature_dict['poi_to_poi_radius_counts'] is not None: 
				temp_feature_list2 = closest_pois_counts_per_label[poi_id]
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None:
				temp_feature_list3 = closest_pois_boolean_per_label_streets[poi_id]
			if feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 
				temp_feature_list4 = closest_pois_counts_per_label_streets[poi_id]
			if feature_dict['poi_to_poi_neighbors_boolean'] is not None:
				temp_feature_list5 = poi_to_poi_neighbors_boolean[poi_id]
			if feature_dict['poi_to_poi_neighbors_counts'] is not None:
				temp_feature_list6 = poi_to_poi_neighbors_counts[poi_id]
		

			"""
			if count == 0:
				print("poi id to nearby poi labels: {0}".format(len(temp_feature_list1)))
				print("poi id to nearby street labels: {0}".format(len(temp_feature_list2)))
				print("poi id to class centroid similarities: {0}".format(len(poi_id_to_class_centroid_similarities_train[poi_id])))
				print("poi id to roken features: {0}".format(len(poi_id_to_word_features[poi_id])))
				print("poi id to n-gram token features: {0}".format(len(poi_id_to_word_features_ngrams_tokens[poi_id])))
				print("poi id to n-gram features: {0}".format(len(poi_id_to_word_features_ngrams[poi_id])))
				
			count += 1
			"""			
			if feature_dict['class_centroid_similarities'] is not None:
				feature_dict['class_centroid_similarities'].append(poi_id_to_class_centroid_similarities_test[poi_id])
			if feature_dict['word_features'] is not None:
				feature_dict['word_features'].append(poi_id_to_word_features[poi_id])
			if feature_dict['word_features_ngrams'] is not None:
				feature_dict['word_features_ngrams'].append(poi_id_to_word_features_ngrams[poi_id])
			if feature_dict['word_features_ngrams_tokens'] is not None:
				feature_dict['word_features_ngrams_tokens'].append(poi_id_to_word_features_ngrams_tokens[poi_id])
			if feature_dict['poi_to_poi_radius_counts'] is not None:
				feature_dict['poi_to_poi_radius_counts'].append(temp_feature_list1)
			if feature_dict['poi_to_poi_radius_boolean'] is not None: 
				feature_dict['poi_to_poi_radius_boolean'].append(temp_feature_list2)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None: 
				feature_dict['poi_to_closest_street_to_poi_radius_boolean'].append(temp_feature_list3)
			if feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 
				feature_dict['poi_to_closest_street_to_poi_radius_counts'].append(temp_feature_list4)
			if feature_dict['poi_to_poi_neighbors_boolean'] is not None: 
				feature_dict['poi_to_poi_neighbors_boolean'].append(temp_feature_list5)
			if feature_dict['poi_to_poi_neighbors_counts'] is not None: 
				feature_dict['poi_to_poi_neighbors_counts'].append(temp_feature_list6)
			
			#feature_list = poi_id_to_class_centroid_similarities_train[poi_id] + poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + poi_id_to_word_features_ngrams_tokens[poi_id] + temp_feature_list1 + temp_feature_list2
			#X_train.append(feature_list)
		sentinel = 0
		for key in feature_dict:
			if feature_dict[key] is not None:
				if sentinel == 0:
					X_test = np.asarray(feature_dict[key])
					if key in config.initialConfig.features_to_normalize:
						X_test = standardize_data_test(X_test, scaler_dict[key])
					if fold_number is not None:
						filepath = folderpath + '/' + key + '_' + 'test_' + str(fold_number) + '.csv'
						np.savetxt(filepath, X_test, delimiter=",")
					else:
						filepath = folderpath + '/' + key + '_' + 'test' + '.csv'
						np.savetxt(filepath, X_test, delimiter=",")
					sentinel = 1
				else:
					temp_array = np.asarray(feature_dict[key])
					if key in config.initialConfig.features_to_normalize:
						temp_array = standardize_data_test(temp_array, scaler_dict[key])
					if fold_number is not None:
						filepath = folderpath + '/' + key + '_' + 'test_' + str(fold_number) + '.csv'
						np.savetxt(filepath, temp_array, delimiter=",")
					else:
						filepath = folderpath + '/' + key + '_' + 'test' + '.csv'
						np.savetxt(filepath, temp_array, delimiter=",")
					X_test = np.concatenate((X_test, np.asarray(feature_dict[key])), axis = 1)
					#sentinel = 1

		#X_train = np.asarray(X_train)
		#X_test = np.asarray(X_test)
	
	#print(X_train.shape)
	#print(X_test.shape)
		
	return X_train, y_train, X_test, y_test
	
def get_test_set(conn, args, poi_ids):
	
	from sklearn.feature_selection import VarianceThreshold
	sel = VarianceThreshold(threshold = (0.2))
	
	#print(len(poi_ids))
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	if args["pois_tbl_name"] is not None:
		poi_df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict(conn, args)
	else:
		poi_df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict_csv(conn, args)
		
	#print(class_codes_set)
		
	# we encode them so we can have a more compact representation of them
	label_encoder, poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict_csv(args, None, poi_id_to_class_code_coordinates_dict)
	args['label_encoder'] = label_encoder
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
	
	feature_dict = dict((el, None) for el in config.initialConfig.feature_list)
	for key in config.initialConfig.included_features:
		feature_dict[key] = []

	scaler_dict = dict((el, None) for el in config.initialConfig.included_features)
	
	if feature_dict['class_centroid_similarities'] is not None:
		poi_id_to_class_centroid_similarities_train, encoded_labels_corpus_train = get_poi_id_to_class_centroid_similarities(poi_ids, poi_id_to_class_code_coordinates_dict, encoded_labels_set, conn, args, [])
	if feature_dict['word_features_ngrams'] is not None:
		poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids, conn, args, config.initialConfig.top_k_character_ngrams_percentage)
	if feature_dict['word_features'] is not None:
		poi_id_to_word_features = get_features_top_k(poi_ids, conn, args, config.initialConfig.top_k_terms_percentage)
	if feature_dict['word_features_ngrams_tokens'] is not None:
		poi_id_to_word_features_ngrams_tokens = get_features_top_k_ngrams_tokens(poi_ids, conn, args, config.initialConfig.top_k_terms_percentage)
	
	if args["pois_tbl_name"] is not None:
		if feature_dict['poi_to_poi_radius_boolean'] is not None or feature_dict['poi_to_poi_radius_counts'] is not None: 
			closest_pois_boolean_per_label, closest_pois_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
		if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None or feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 	
			closest_pois_boolean_per_label_streets, closest_pois_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
	else:
		if feature_dict['poi_to_poi_radius_boolean'] is not None or feature_dict['poi_to_poi_radius_counts'] is not None: 
			closest_pois_boolean_per_label, closest_pois_counts_per_label = get_closest_pois_boolean_and_counts_per_label_csv(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
		if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None or feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 	
			closest_pois_boolean_per_label_streets, closest_pois_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets_csv(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
	
	if feature_dict['poi_to_poi_neighbors_boolean'] is not None or feature_dict['poi_to_poi_neighbors_counts'] is not None:
		poi_to_poi_neighbors_boolean, poi_to_poi_neighbors_counts = get_neighbor_pois_boolean_and_counts_per_label_csv(poi_ids, conn, args)	
	
	for poi_id in poi_ids:
		if feature_dict['poi_to_poi_radius_boolean'] is not None:
			temp_feature_list1 = closest_pois_boolean_per_label[poi_id]
		if feature_dict['poi_to_poi_radius_counts'] is not None: 
			temp_feature_list2 = closest_pois_counts_per_label[poi_id]
		if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None:
			temp_feature_list3 = closest_pois_boolean_per_label_streets[poi_id]
		if feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 
			temp_feature_list4 = closest_pois_counts_per_label_streets[poi_id]
		if feature_dict['poi_to_poi_neighbors_boolean'] is not None:
			temp_feature_list5 = poi_to_poi_neighbors_boolean[poi_id]
		if feature_dict['poi_to_poi_neighbors_boolean'] is not None:
			temp_feature_list6 = poi_to_poi_neighbors_counts[poi_id]
		
		"""
		if count == 0:
			print("poi id to nearby poi labels: {0}".format(len(temp_feature_list1)))
			print("poi id to nearby street labels: {0}".format(len(temp_feature_list2)))
			print("poi id to class centroid similarities: {0}".format(len(poi_id_to_class_centroid_similarities_train[poi_id])))
			print("poi id to roken features: {0}".format(len(poi_id_to_word_features[poi_id])))
			print("poi id to n-gram token features: {0}".format(len(poi_id_to_word_features_ngrams_tokens[poi_id])))
			print("poi id to n-gram features: {0}".format(len(poi_id_to_word_features_ngrams[poi_id])))
			
		count += 1
		"""
		if feature_dict['class_centroid_similarities'] is not None:
			feature_dict['class_centroid_similarities'].append(poi_id_to_class_centroid_similarities_train[poi_id])
		if feature_dict['word_features'] is not None:
			feature_dict['word_features'].append(poi_id_to_word_features[poi_id])
		if feature_dict['word_features_ngrams'] is not None:
			feature_dict['word_features_ngrams'].append(poi_id_to_word_features_ngrams[poi_id])
		if feature_dict['word_features_ngrams_tokens'] is not None:
			feature_dict['word_features_ngrams_tokens'].append(poi_id_to_word_features_ngrams_tokens[poi_id])
		if feature_dict['poi_to_poi_radius_boolean'] is not None:
			feature_dict['poi_to_poi_radius_boolean'].append(temp_feature_list1)
		if feature_dict['poi_to_poi_radius_counts'] is not None: 
			feature_dict['poi_to_poi_radius_counts'].append(temp_feature_list2)
		if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None: 
			feature_dict['poi_to_closest_street_to_poi_radius_boolean'].append(temp_feature_list3)
		if feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 
			feature_dict['poi_to_closest_street_to_poi_radius_counts'].append(temp_feature_list4)
		if feature_dict['poi_to_poi_neighbors_boolean'] is not None: 
			feature_dict['poi_to_poi_neighbors_boolean'].append(temp_feature_list5)
		if feature_dict['poi_to_poi_neighbors_counts'] is not None: 
			feature_dict['poi_to_poi_neighbors_counts'].append(temp_feature_list6)
		
		#feature_list = poi_id_to_class_centroid_similarities_train[poi_id] + poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + poi_id_to_word_features_ngrams_tokens[poi_id] + temp_feature_list1 + temp_feature_list2
		#X_train.append(feature_list)
		
	sentinel = 0
	for key in feature_dict:
		if feature_dict[key] is not None:
			if sentinel == 0:
				#print("edw1")
				#print(key, feature_dict[key])
				X_train = np.asarray(feature_dict[key])
				#print(key, X_train)
				if key in config.initialConfig.features_to_normalize:
					X_train, scaler_dict[key] = standardize_data_train(X_train)
				
				print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(X_train), np.std(X_train), np.amax(X_train), np.amin(X_train), X_train.shape))
				
				#X_train = sel.fit_transform(X_train)
				#print("Results after removal of features with low variance:")
				#print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(X_train), np.std(X_train), np.amax(X_train), np.amin(X_train), X_train.shape))
				
				sentinel = 1
			else:
				#print("edw1")
				#print(key, feature_dict[key])
				temp_array = np.asarray(feature_dict[key])
				#print(key, temp_array)
				if key in config.initialConfig.features_to_normalize:
					temp_array, scaler_dict[key] = standardize_data_train(temp_array)
				
				print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(temp_array), np.std(temp_array), np.amax(temp_array), np.amin(temp_array), temp_array.shape))
				#X_train = sel.fit_transform(X_train)
				#print("Results after removal of features with low variance:")
				#print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(X_train), np.std(X_train), np.amax(X_train), np.amin(X_train), X_train.shape))
				X_train = np.concatenate((X_train, temp_array), axis = 1)
				
				#sentinel = 1	
		
	return X_train, y_train
	
def get_train_set(conn, args, poi_ids):
	
	from sklearn.feature_selection import VarianceThreshold
	sel = VarianceThreshold(threshold = (0.2))
	
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
	label_encoder, poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict_csv(args, class_codes_set, poi_id_to_class_code_coordinates_dict)
	args['label_encoder'] = label_encoder
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
	
	feature_dict = dict((el, None) for el in config.initialConfig.feature_list)
	for key in config.initialConfig.included_features:
		feature_dict[key] = []
		
	sentinel = 0

	for key in config.initialConfig.included_features:
		if config.initialConfig.experiment_folder == None:
			experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
			list_of_folders = glob.glob(experiment_folder_path)
			if list_of_folders == []:
				print("ERROR! No experiment folder found inside the root folder")
				return
			else:
				latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
		else:
			latest_experiment_folder = config.initialConfig.root_path + config.initialConfig.experiment_folder
			
		filepath = latest_experiment_folder + '/' + key + '_' + 'model_training_' + str(args['level']) + '.csv'
		exists = os.path.isfile(filepath)
		if exists:
			if sentinel == 0:
				X_train = np.genfromtxt(filepath, delimiter=',')
				sentinel = 1
			else:
				temp_array = np.genfromtxt(filepath, delimiter=',')
				X_train = np.concatenate((X_train, temp_array), axis = 1)
				#sentinel = 1
	
	if not exists:
		scaler_dict = dict((el, None) for el in config.initialConfig.included_features)
		
		if feature_dict['class_centroid_similarities'] is not None:
			poi_id_to_class_centroid_similarities_train, encoded_labels_corpus_train = get_poi_id_to_class_centroid_similarities(poi_ids, poi_id_to_class_code_coordinates_dict, encoded_labels_set, conn, args, [])
		if feature_dict['word_features_ngrams'] is not None:
			poi_id_to_word_features_ngrams = get_features_top_k_ngrams(poi_ids, conn, args, config.initialConfig.top_k_character_ngrams_percentage)
		if feature_dict['word_features'] is not None:
			poi_id_to_word_features = get_features_top_k(poi_ids, conn, args, config.initialConfig.top_k_terms_percentage)
		if feature_dict['word_features_ngrams_tokens'] is not None:
			poi_id_to_word_features_ngrams_tokens = get_features_top_k_ngrams_tokens(poi_ids, conn, args, config.initialConfig.top_k_terms_percentage)
		
		if args["pois_tbl_name"] is not None:
			if feature_dict['poi_to_poi_radius_boolean'] is not None or feature_dict['poi_to_poi_radius_counts'] is not None: 
				closest_pois_boolean_per_label, closest_pois_counts_per_label = get_closest_pois_boolean_and_counts_per_label(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None or feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 	
				closest_pois_boolean_per_label_streets, closest_pois_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
		else:
			if feature_dict['poi_to_poi_radius_boolean'] is not None or feature_dict['poi_to_poi_radius_counts'] is not None: 
				closest_pois_boolean_per_label, closest_pois_counts_per_label = get_closest_pois_boolean_and_counts_per_label_csv(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None or feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 	
				closest_pois_boolean_per_label_streets, closest_pois_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets_csv(poi_ids, conn, args, config.initialConfig.threshold_distance_neighbor_pois_roads)
		
		if feature_dict['poi_to_poi_neighbors_boolean'] is not None or feature_dict['poi_to_poi_neighbors_counts'] is not None:
			poi_to_poi_neighbors_boolean, poi_to_poi_neighbors_counts = get_neighbor_pois_boolean_and_counts_per_label_csv(poi_ids, conn, args)	
		
		for poi_id in poi_ids:
			if feature_dict['poi_to_poi_radius_boolean'] is not None:
				temp_feature_list1 = closest_pois_boolean_per_label[poi_id]
			if feature_dict['poi_to_poi_radius_counts'] is not None: 
				temp_feature_list2 = closest_pois_counts_per_label[poi_id]
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None:
				temp_feature_list3 = closest_pois_boolean_per_label_streets[poi_id]
			if feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 
				temp_feature_list4 = closest_pois_counts_per_label_streets[poi_id]
			if feature_dict['poi_to_poi_neighbors_boolean'] is not None:
				temp_feature_list5 = poi_to_poi_neighbors_boolean[poi_id]
			if feature_dict['poi_to_poi_neighbors_boolean'] is not None:
				temp_feature_list6 = poi_to_poi_neighbors_counts[poi_id]
			
			"""
			if count == 0:
				print("poi id to nearby poi labels: {0}".format(len(temp_feature_list1)))
				print("poi id to nearby street labels: {0}".format(len(temp_feature_list2)))
				print("poi id to class centroid similarities: {0}".format(len(poi_id_to_class_centroid_similarities_train[poi_id])))
				print("poi id to roken features: {0}".format(len(poi_id_to_word_features[poi_id])))
				print("poi id to n-gram token features: {0}".format(len(poi_id_to_word_features_ngrams_tokens[poi_id])))
				print("poi id to n-gram features: {0}".format(len(poi_id_to_word_features_ngrams[poi_id])))
				
			count += 1
			"""
			if feature_dict['class_centroid_similarities'] is not None:
				feature_dict['class_centroid_similarities'].append(poi_id_to_class_centroid_similarities_train[poi_id])
			if feature_dict['word_features'] is not None:
				feature_dict['word_features'].append(poi_id_to_word_features[poi_id])
			if feature_dict['word_features_ngrams'] is not None:
				feature_dict['word_features_ngrams'].append(poi_id_to_word_features_ngrams[poi_id])
			if feature_dict['word_features_ngrams_tokens'] is not None:
				feature_dict['word_features_ngrams_tokens'].append(poi_id_to_word_features_ngrams_tokens[poi_id])
			if feature_dict['poi_to_poi_radius_boolean'] is not None:
				feature_dict['poi_to_poi_radius_boolean'].append(temp_feature_list1)
			if feature_dict['poi_to_poi_radius_counts'] is not None: 
				feature_dict['poi_to_poi_radius_counts'].append(temp_feature_list2)
			if feature_dict['poi_to_closest_street_to_poi_radius_boolean'] is not None: 
				feature_dict['poi_to_closest_street_to_poi_radius_boolean'].append(temp_feature_list3)
			if feature_dict['poi_to_closest_street_to_poi_radius_counts'] is not None: 
				feature_dict['poi_to_closest_street_to_poi_radius_counts'].append(temp_feature_list4)
			if feature_dict['poi_to_poi_neighbors_boolean'] is not None: 
				feature_dict['poi_to_poi_neighbors_boolean'].append(temp_feature_list5)
			if feature_dict['poi_to_poi_neighbors_counts'] is not None: 
				feature_dict['poi_to_poi_neighbors_counts'].append(temp_feature_list6)
			
			#feature_list = poi_id_to_class_centroid_similarities_train[poi_id] + poi_id_to_word_features[poi_id] + poi_id_to_word_features_ngrams[poi_id] + poi_id_to_word_features_ngrams_tokens[poi_id] + temp_feature_list1 + temp_feature_list2
			#X_train.append(feature_list)
		
		sentinel = 0
		for key in feature_dict:
			if feature_dict[key] is not None:
				if sentinel == 0:
					#print("edw1")
					#print(key, feature_dict[key])
					X_train = np.asarray(feature_dict[key])
					#print(key, X_train)
					if key in config.initialConfig.features_to_normalize:
						X_train, scaler_dict[key] = standardize_data_train(X_train)
					
					filepath = args['folderpath'] + '/' + key + '_' + 'model_training_' + str(args['level']) + '.csv'
					np.savetxt(filepath, X_train, delimiter=",")
					
					#print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(X_train), np.std(X_train), np.amax(X_train), np.amin(X_train), X_train.shape))
					
					#X_train = sel.fit_transform(X_train)
					#print("Results after removal of features with low variance:")
					print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(X_train), np.std(X_train), np.amax(X_train), np.amin(X_train), X_train.shape))
					
					sentinel = 1
				else:
					#print("edw1")
					#print(key, feature_dict[key])
					temp_array = np.asarray(feature_dict[key])
					#print(key, temp_array)
					if key in config.initialConfig.features_to_normalize:
						temp_array, scaler_dict[key] = standardize_data_train(temp_array)
					
					#print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(temp_array), np.std(temp_array), np.amax(temp_array), np.amin(temp_array), temp_array.shape))
					#X_train = sel.fit_transform(X_train)
					#print("Results after removal of features with low variance:")
					print("Feature Name: {0}, Mean Value: {1}, Std Value: {2}, Max Value: {3}, Min Value: {4}, Shape: {5}".format(key, np.mean(X_train), np.std(X_train), np.amax(X_train), np.amin(X_train), X_train.shape))
					
					filepath = args['folderpath'] + '/' + key + '_' + 'model_training_' + str(args['level']) + '.csv'
					np.savetxt(filepath, temp_array, delimiter=",")
					
					X_train = np.concatenate((X_train, temp_array), axis = 1)
					
					#sentinel = 1	
		
	return X_train, y_train
	

def standardize_data_train(X):
	from sklearn.preprocessing import MinMaxScaler
	
	standard_scaler = MinMaxScaler()
	X = standard_scaler.fit_transform(X)
	
	return X, standard_scaler

def standardize_data_test(X, scaler):
	from sklearn.preprocessing import MinMaxScaler
	
	X = scaler.transform(X)
	
	return X
	
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
	args['level'] = 1
	args['step'] = 1
	
	if config.initialConfig.experiment_folder == None:
		folderpath = config.initialConfig.root_path + 'experiment_folder_' + str(datetime.datetime.now())
		folderpath = folderpath.replace(':', '-')
		os.makedirs(folderpath)
	else:
		folderpath = config.initialConfig.experiment_folder
		
	args['folderpath'] = folderpath
		
	conn = None
	poi_ids = get_poi_ids(conn, args)
	poi_ids = poi_ids[config.initialConfig.poi_id]
	random.shuffle(poi_ids)	
	poi_ids = list(poi_ids)
	X_train, _ = get_train_set(conn, args, poi_ids)

if __name__ == "__main__":
   main()
