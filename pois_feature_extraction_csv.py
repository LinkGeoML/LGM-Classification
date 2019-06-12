#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import pandas as pd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *
import osmnx as ox
import shapely
import config
from scipy import spatial
import pickle
import dill
import json
import glob
import os

def get_poi_id_to_neighbors_boolean_and_counts_per_class_dict_csv(ids, conn, num_of_labels, poi_id_to_encoded_labels_dict, threshold, args):

	"""
	This function is responsible for mapping the pois to two lists of size equal
	to the number of available classes in the dataset. The first list will contain
	boolean values referring to whether a poi of that corresponding class is among
	the k nearest neighbors of the poi of interest. The second list will contain
	numerical values referring the the total number of pois of that corresponding class
	that are among the k nearest neighbors of the poi of interest. 
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within the k nearest neighbors of the poi with id = 1, then the dictionaries will look like this: 
	poi_id_to_label_boolean_dict[1] = [1, 0, 1]
	poi_id_to_label_counts_dict[1] = [2, 0, 3]
	
	Arguments
	---------
	ids: :obj:`list`
		the ids of the pois which we want to be contained in the dictionary keys
	num_of_labels: :obj:`int`
		the total number of the different labels in the dataset
	encoded_labels_id_dict: :obj:`dictionary`
		the dictionary mapping the poi ids to labels
	threshold: the aforementioned threshold (redundant)
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	
	Returns
	-------
	poi_id_to_label_boolean_dict: :obj:`dictionary`
		contains boolean values referring to whether a poi of that 
		corresponding class is among the k nearest neighbors of the poi of interest
	poi_id_to_label_counts_dict: :obj:`dictionary`
		contains numerical values referring the the total number 
		of pois of that corresponding class that are among the k nearest neighbors of the poi of interest
	"""

	from sklearn.neighbors import DistanceMetric
	from scipy.spatial import distance

	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')

	df = pd.read_csv(args['pois_csv_name'])

	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
	
	poi_id_to_label_boolean_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	poi_id_to_label_counts_dict = dict.fromkeys(df[config.initialConfig.poi_id])

	# add dummy values to the dictionary in order to initialize it
	# in a form that resembles its desired final form
	for poi_id in poi_id_to_label_boolean_dict:
		poi_id_to_label_boolean_dict[poi_id] = [0 for _ in range(0, num_of_labels)]
		
	for poi_id in poi_id_to_label_counts_dict:
		poi_id_to_label_counts_dict[poi_id] = [0 for _ in range(0, num_of_labels)]


	x = np.asarray(df[config.initialConfig.x])
	y = np.asarray(df[config.initialConfig.y])
	poi_coordinate_array = np.stack((x, y), axis = -1)
	spatial_index = spatial.KDTree(poi_coordinate_array)

	for index, row in df.iterrows():
		pois_within_radius = spatial_index.query((row[config.initialConfig.x], row[config.initialConfig.y]), config.initialConfig.num_poi_neighbors)
		pois_within_radius = list(pois_within_radius[1])
		index = int(index)
		if index in pois_within_radius:
			pois_within_radius.remove(index)
		
		if pois_within_radius is not None:
			for poi_index in pois_within_radius:
				poi_id_to_label_boolean_dict[row[config.initialConfig.poi_id]][poi_id_to_encoded_labels_dict[df.iloc[poi_index][config.initialConfig.poi_id]][0][0]] = 1
				poi_id_to_label_counts_dict[row[config.initialConfig.poi_id]][poi_id_to_encoded_labels_dict[df.iloc[poi_index][config.initialConfig.poi_id]][0][0]] += 1

	return poi_id_to_label_boolean_dict, poi_id_to_label_counts_dict

def get_poi_id_to_closest_poi_ids_dict_csv(ids, conn, args):
	
	"""
	This function is responsible for mapping each poi to a list of pois. Each list will contain
	the ids of the pois that are situated within threshold distance from the poi of interest.
	
	For example, if three pois with ids 1,5,7 respectively are situated within threshold distance 
	of the poi with id = 3, then the dictionary will look like this: 
	poi_id_to_closet_poi_ids_dict[3] = [1,5,7]
	
	Arguments
	---------
	ids: :obj:`list`
		the ids of the pois which we want to be contained in the dictionary keys
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	
	Returns
	-------
	poi_id_to_closet_poi_ids_dict: :obj:`dictionary`
		dictionary that maps each poi id to a list of poi ids 
		that belong to the pois that are situated within threshold distance from the poi of interest.
	"""
	
	from sklearn.neighbors import DistanceMetric
	from scipy.spatial import distance
	
	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')

	df = pd.read_csv(args['pois_csv_name'])
	if args['level'] == 1:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[0]: 'class_code'})
	elif args['level'] == 2:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[1]: 'class_code'})
	else:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[2]: 'class_code'})
	
	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
	
	poi_id_to_closet_poi_ids_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	
	# add dummy values to the dictionary in order to initialize it
	# in a form that resembles its desired final form
	for poi_id in poi_id_to_closet_poi_ids_dict:
		poi_id_to_closet_poi_ids_dict[poi_id] = []
	
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			if args['step'] == 1 or args['step'] == 2:
				index_folderpath = latest_experiment_folder + '/' + 'pois_index_' + str(args['fold_number'])
			else:
				index_folderpath = latest_experiment_folder + '/' + 'pois_index_train'
			exists = os.path.exists(index_folderpath)
			if not exists:
				os.mkdir(index_folderpath)
				x = np.asarray(df[config.initialConfig.x])
				y = np.asarray(df[config.initialConfig.y])
				poi_coordinate_array = np.stack((x, y), axis = -1)
				spatial_index = spatial.KDTree(poi_coordinate_array)
				index_filepath = index_folderpath + '/' + 'pois_index.pkl'
				with open(index_filepath, 'wb') as fdump:
					pickle.dump(spatial_index, fdump)
			else:
				index_filepath = index_folderpath + '/' + 'pois_index.pkl'
				pickle_in = open(index_filepath, "rb")
				spatial_index = pickle.load(pickle_in)
	else:
		if args['step'] == 1 or args['step'] == 2:
			index_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + 'pois_index_' + str(args['fold_number']) + '/' + 'pois_index.pkl'
		else:
			index_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + 'pois_index_train' + '/' + 'pois_index.pkl'
		pickle_in = open(index_filepath, "rb")
		spatial_index = pickle.load(pickle_in)
	for index, row in df.iterrows():
		pois_within_radius = spatial_index.query_ball_point((row[config.initialConfig.x], row[config.initialConfig.y]), config.initialConfig.threshold_distance_neighbor_pois_roads)
		if index in pois_within_radius:
			pois_within_radius.remove(int(index))
		for poi_index in pois_within_radius:
			poi_id_to_closet_poi_ids_dict[row[config.initialConfig.poi_id]].append(poi_index)
	return poi_id_to_closet_poi_ids_dict
	
def get_poi_id_to_closest_street_id_dict_csv(ids, conn, args):
	
	"""
	This function is responsible for mapping each poi to a list of streets and each street
	to a list of pois. Each list will contain the ids of the pois (or streets) that are situated 
	within threshold distance from the poi (or street) of interest.
	
	Arguments
	---------
	ids: :obj:`list`
		the ids of the pois which we want to be contained in the dictionary keys (or values)
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	
	Returns
	-------
	poi_id_to_street_geom_dict: :obj:`dictionary`
		dictionary that maps each poi id to a list of street geometries
		that belong to the streets that are situated within threshold distance from the poi of interest.
	
	street_geom_to_to_closest_poi_ids_dict: :obj:`dictionary`
		dictionary that maps each street geometry to a list of 
		poi ids that belong to those pois that are situated within threshold distance from the street of 
		interest.
	"""
	
	from sklearn.neighbors import DistanceMetric
	from scipy.spatial import distance
	
	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')
		
	from functools import partial
	import pyproj
	from shapely.ops import transform
	
	from shapely.geometry import Point

	project = partial(
		pyproj.transform,
		pyproj.Proj(init='epsg:2100'), # source coordinate system
		pyproj.Proj(init='epsg:4326')) # destination coordinate system
	
	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')
	
	poi_df = pd.read_csv(args['pois_csv_name'])
	if args['level'] == 1:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[0]: 'class_code'})
	elif args['level'] == 2:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[1]: 'class_code'})
	else:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[2]: 'class_code'})
	
	is_in_ids = []
	all_ids = poi_df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	poi_df = poi_df[is_in_ids]
	
	# construct a dictionary from their ids
	# also get its class_code
	poi_id_to_street_geom_dict = dict.fromkeys(poi_df[config.initialConfig.poi_id])
	for poi_id in poi_id_to_street_geom_dict:
		poi_id_to_street_geom_dict[poi_id] = [0, 0, 0]
	
	if config.initialConfig.osmnx_bbox:
		g = ox.graph_from_bbox(north = config.initialConfig.osmnx_bbox_coordinates[0], 
		south = config.initialConfig.osmnx_bbox_coordinates[1],
		east = config.initialConfig.osmnx_bbox_coordinates[2],
		west = config.initialConfig.osmnx_bbox_coordinates[3])
	else:
		g = ox.graph_from_place(config.initialConfig.osmnx_placename)
	gdf_nodes, street_df = ox.graph_to_gdfs(g)
	
	geom_keys = []
	for index, row in street_df.iterrows():
		geom_keys.append(str(row['geometry']))
	street_geom_to_to_closest_poi_ids_dict = dict.fromkeys(geom_keys)
	for street_geom in street_geom_to_to_closest_poi_ids_dict:
		street_geom_to_to_closest_poi_ids_dict[street_geom] = []
	
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			if args['step'] == 1 or args['step'] == 2:
				index_folderpath = latest_experiment_folder + '/' + 'street_df_' + str(args['fold_number'])
			else:
				index_folderpath = latest_experiment_folder + '/' + 'street_df_train'
			exists = os.path.exists(index_folderpath)
			if not exists:
				os.mkdir(index_folderpath)
				spatial_index = street_df.sindex
				index_filepath = index_folderpath + '/' + 'street_df.pkl'
				street_df.to_pickle(index_filepath)
			else:
				spatial_index = street_df.sindex
	else:
		spatial_index = street_df.sindex
	
	if config.initialConfig.experiment_folder == None:
		experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
		list_of_folders = glob.glob(experiment_folder_path)
		if list_of_folders == []:
			print("ERROR! No experiment folder found inside the root folder")
			return
		else:
			latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
			if args['step'] == 1 or args['step'] == 2:
				index_folderpath = latest_experiment_folder + '/' + 'street_geom_to_poi_id_dict_' + str(args['fold_number'])
			else:
				index_folderpath = latest_experiment_folder + '/' + 'street_geom_to_poi_id_dict_train'
			exists = os.path.exists(index_folderpath)
			if not exists:
				for index, row in poi_df.iterrows():
					poi_geom = Point(row[config.initialConfig.x], row[config.initialConfig.y])
					poi_geom = transform(project, poi_geom)
					
					coords = list(poi_geom.coords)[0]
					nearest_street_geom_index = list(spatial_index.nearest(coords, num_results = 1))
					nearest_street_geom = street_df.iloc[nearest_street_geom_index[0]]['geometry']
					poi_id_to_street_geom_dict[row[config.initialConfig.poi_id]] = str(nearest_street_geom)
					street_geom_to_to_closest_poi_ids_dict[str(nearest_street_geom)].append([row[config.initialConfig.poi_id], row['class_code']])

				os.mkdir(index_folderpath)
				json_filepath = index_folderpath + '/' + 'street_geom_to_to_closest_poi_ids_dict.json'
				with open(json_filepath, 'w') as fp:
					json.dump(street_geom_to_to_closest_poi_ids_dict, fp)
			else:
				for index, row in poi_df.iterrows():
					poi_geom = Point(row[config.initialConfig.x], row[config.initialConfig.y])
					poi_geom = transform(project, poi_geom)
										
					coords = list(poi_geom.coords)[0]
					nearest_street_geom_index = list(spatial_index.nearest(coords, num_results = 1))
					nearest_street_geom = street_df.iloc[nearest_street_geom_index[0]]['geometry']
					poi_id_to_street_geom_dict[row[config.initialConfig.poi_id]] = str(nearest_street_geom)
					
				json_filepath = index_folderpath + '/' + 'street_geom_to_to_closest_poi_ids_dict.json'
				with open(json_filepath, 'r') as f:
					street_geom_to_to_closest_poi_ids_dict = json.load(f)
	else:
		for index, row in poi_df.iterrows():
			poi_geom = Point(row[config.initialConfig.x], row[config.initialConfig.y])
			poi_geom = transform(project, poi_geom)
						
			coords = list(poi_geom.coords)[0]
			nearest_street_geom_index = list(spatial_index.nearest(coords, num_results = 1))
			nearest_street_geom = street_df.iloc[nearest_street_geom_index[0]]['geometry']
			poi_id_to_street_geom_dict[row[config.initialConfig.poi_id]] = str(nearest_street_geom)
		
		if args['step'] == 1 or args['step'] == 2:
			json_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + 'street_geom_to_poi_id_dict_' + str(args['fold_number']) + '/' + 'street_geom_to_to_closest_poi_ids_dict.json'
		else:
			json_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/' + 'street_geom_to_poi_id_dict_train' + '/' + 'street_geom_to_to_closest_poi_ids_dict.json'
		with open(json_filepath, 'r') as f:
			street_geom_to_to_closest_poi_ids_dict = json.load(f)

	return poi_id_to_street_geom_dict, street_df, street_geom_to_to_closest_poi_ids_dict
	
def get_street_geom_to_closest_poi_ids_dict_csv(ids, conn, args, street_df):
	
	"""
	This function is responsible for mapping each street geometry to a list containing poi ids.
	Each mapping will contain the ids of the pois that are situated within threshold distance from
	the street geometry of interest.
	
	Arguments
	---------
	ids: :obj:`list`
		the ids of the pois which we want to be contained in the dictionary keys (or values)
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	street_df: :obj:`pandas dataframe`
		dataframe containing street info
	
	Returns
	-------
	street_geom_to_to_closest_poi_ids_dict: :obj:`dictionary`
		dictionary that maps each street geometry
		to a list containing poi ids that correspond to those pois that are situated within
		threshold distance from the street of interest.
	"""
	
	from sklearn.neighbors import DistanceMetric
	from scipy.spatial import distance
	
	from functools import partial
	import pyproj
	from shapely.ops import transform
	
	from shapely.geometry import Point

	project = partial(
		pyproj.transform,
		pyproj.Proj(init=config.initialConfig.original_SRID), # source coordinate system
		pyproj.Proj(init='epsg:4326')) # destination coordinate system
	
	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')
	
	poi_df = pd.read_csv(args['pois_csv_name'])
	if args['level'] == 1:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[0]: 'class_code'})
	elif args['level'] == 2:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[1]: 'class_code'})
	else:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[2]: 'class_code'})
	
	is_in_ids = []
	all_ids = poi_df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	poi_df = poi_df[is_in_ids]
	
	geom_keys = []
	for index, row in street_df.iterrows():
		geom_keys.append(str(row['geometry']))
	street_geom_to_to_closest_poi_ids_dict = dict.fromkeys(geom_keys)
	for street_geom in street_geom_to_to_closest_poi_ids_dict:
		street_geom_to_to_closest_poi_ids_dict[street_geom] = []
	
	for index, row in street_df.iterrows():
		street_geom = row['geometry']
		for index1, row1 in poi_df.iterrows():
			poi_geom = Point(row1[config.initialConfig.x], row1[config.initialConfig.y])
			poi_geom = transform(project, poi_geom)
			if street_geom.distance(poi_geom) * 111000 < config.initialConfig.threshold_distance_neighbor_pois_roads:
				street_geom_to_to_closest_poi_ids_dict[str(street_geom)].append([row1[config.initialConfig.poi_id], row1['class_code']])
	return street_geom_to_to_closest_poi_ids_dict
	
def construct_final_feature_vector_csv(ids, conn, args, num_of_labels, poi_id_to_encoded_labels_dict, poi_id_to_closest_poi_ids_dict, poi_id_to_closest_street_id_dict, street_geom_to_closest_poi_ids_dict):
	
	"""
	This function is responsible for providing two dictionaries. The first dictionary will map each
	poi_id to boolean values that indicate whether pois belonging to each class index are situated
	on the same road and are within threshold distance from its position. The second dictionary will map each
	poi_id to numerical values that indicate the total number of pois belonging to each class index are situated
	on the same road and are within threshold distance from its position.
	
	Arguments
	---------
	ids: :obj:`list`
		a list containing the ids of the pois for which the feature extraction process is to be
		executed
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	num_of_labels: :obj:`int`
		the total number of the different labels
	
	Returns
	-------
	poi_id_to_closest_pois_street_boolean_per_label_dict: :obj:`dictionary`
		the first dictionary as described above
	poi_id_to_closest_pois_street_counts_per_label_dict: :obj:`dictionary`
		the second dictionary as described above
	"""
	
	poi_df = pd.read_csv(args['pois_csv_name'])
	if args['level'] == 1:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[0]: 'class_code'})
	elif args['level'] == 2:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[1]: 'class_code'})
	else:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[2]: 'class_code'})
	
	is_in_ids = []
	all_ids = poi_df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	poi_df = poi_df[is_in_ids]

	# create a dictionary with the poi ids as its keys
	id_dict = dict.fromkeys(poi_df[config.initialConfig.poi_id])
	for index, row in poi_df.iterrows():
		id_dict[row[config.initialConfig.poi_id]] = [row['class_code'], 0, 0]
	
	poi_id_to_closest_pois_street_boolean_per_label_dict = dict.fromkeys(poi_id_to_closest_poi_ids_dict.keys())
	poi_id_to_closest_pois_street_counts_per_label_dict = dict.fromkeys(poi_id_to_closest_poi_ids_dict.keys())
	
	# prepare the street id dictionary to be able to store the
	# boolean and count duplet for each of the class labels
	for poi_id in poi_id_to_closest_pois_street_boolean_per_label_dict:
		poi_id_to_closest_pois_street_boolean_per_label_dict[poi_id] = [0 for _ in range(0, num_of_labels)]
	
	for poi_id in poi_id_to_closest_pois_street_counts_per_label_dict:
		poi_id_to_closest_pois_street_counts_per_label_dict[poi_id] = [0 for _ in range(0, num_of_labels)]

	for poi_id in poi_id_to_closest_street_id_dict:
		street_geometry = poi_id_to_closest_street_id_dict[poi_id]
		for street_neighboring_poi_id_class_code_list in street_geom_to_closest_poi_ids_dict[str(street_geometry)]:
			class_index = args['label_encoder'].transform([street_neighboring_poi_id_class_code_list[1]])
			poi_id_to_closest_pois_street_boolean_per_label_dict[poi_id][class_index[0]] = 1
			poi_id_to_closest_pois_street_counts_per_label_dict[poi_id][class_index[0]] += 1

	return poi_id_to_closest_pois_street_boolean_per_label_dict, poi_id_to_closest_pois_street_counts_per_label_dict
	
def get_closest_pois_boolean_and_counts_per_label_streets_csv(ids, conn, args, threshold = 1000.0):
	
	"""
	This function is responsible for uniting the functionalities of several street-based functions
	in order to provide street-based geospatial features.
	
	Arguments
	---------
	ids: :obj:`list`
		a list containing the ids of the pois for which the feature extraction process is to be
		executed
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	threshold: (redundant)
	
	Returns
	-------
	boolean_feature_vector: :obj:`dictionary`
		refer to the description of construct_final_feature_vector_csv
	counts_feature_vector: :obj:`dictionary`
		refer to the description of construct_final_feature_vector_csv
	"""
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict_csv(conn, args)
	
	# we read the different labels
	class_codes_set = get_class_codes_set_csv(args, df)
		
	# we encode them so we can have a more compact representation of them
	_, poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict_csv(args, class_codes_set, poi_id_to_class_code_coordinates_dict)
	
	poi_id_to_closest_poi_ids_dict = get_poi_id_to_closest_poi_ids_dict_csv(ids, conn, args)
	# get the dictionary mapping each poi id to that of its closest road
	poi_id_to_closest_street_id_dict, street_df, street_geom_to_closest_poi_ids_dict = get_poi_id_to_closest_street_id_dict_csv(ids, conn, args)
	boolean_feature_vector, counts_feature_vector = construct_final_feature_vector_csv(ids, conn, args, len(encoded_labels_set), poi_id_to_encoded_labels_dict, poi_id_to_closest_poi_ids_dict, poi_id_to_closest_street_id_dict, street_geom_to_closest_poi_ids_dict)
	
	return boolean_feature_vector, counts_feature_vector
	
def get_class_codes_set_csv(args, df):
	
	"""
	This function is responsible for returning a set containing the 
	encoded class codes.
	
	Arguments
	---------
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	df: :obj:`pandas dataframe`
		the dataframe containing our data.
	
	Returns
	-------
	class_codes: :obj:`list`
		a list of the class codes
	"""
	
	# read the file containing the class codes
	#df = pd.read_excel('/home/nikos/Desktop/Datasets/GeoData_PoiMarousi/GeoData_poiClasses.xlsx', sheet_name=None)		
	
	# store the class codes (labels) in the list
	if args['step'] < 4:
		class_codes = list(df['class_code'])
		class_codes = list(set(class_codes))
	else:
		if config.initialConfig.experiment_folder == None:
			experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
			list_of_folders = glob.glob(experiment_folder_path)
			if list_of_folders == []:
				print("ERROR! No experiment folder found inside the root folder")
				return
			else:
				latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
				descriptor_filepath = latest_experiment_folder + '/label_count.txt'
				with open(descriptor_filepath, 'r') as f:
					encoded_labels_count = f.read()
					encoded_labels_list = [i for i in range(0, int(encoded_labels_count))]
					class_codes = list(set(encoded_labels_list))
		else:
			descriptor_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/label_count.txt'
			with open(descriptor_filepath, 'r') as f:
				encoded_labels_count = f.read()
				encoded_labels_list = [i for i in range(0, int(encoded_labels_count))]
				class_codes = list(set(encoded_labels_list))

	return class_codes
	
def get_poi_id_to_encoded_labels_dict_csv(args, labels_set, id_dict):
	
	"""
	This function is responsible for mapping the poi ids to the labels that
	correspond to the class that each poi identified by that poi id belongs
	It encodes the original labels to values between 0 and len(labels_set) in order 
	to have a more compact and user-friendly encoding of them.
	
	Arguments
	---------
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	labels_set: :obj:`list`
		the original labels (not encoded)
	id_dict: :obj:`dictionary`
		a dictionary that has the poi ids as its keys
	
	Returns
	-------
	
	le: :obj:`scikit-learn label encoder object`
		the instance of the label encoder used so that
		future use of him can be made possible
	
	id_dict: :obj:`dictionary`
		an updated version of our pois dictionary
		now mapping their ids to their encoded labels
	
	labels_set: :obj:`set`
		the encoded labels set
	"""
	
	from sklearn.preprocessing import LabelEncoder
	
	if args['step'] < 4:
		# fit the label encoder to our labels set
		le = LabelEncoder()
		le.fit(labels_set)
		
		if config.initialConfig.experiment_folder == None:
			experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
			list_of_folders = glob.glob(experiment_folder_path)
			if list_of_folders == []:
				print("ERROR! No experiment folder found inside the root folder")
				return
			else:
				latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
				descriptor_filepath = latest_experiment_folder + '/classes.pkl'
				output = open(descriptor_filepath, 'wb')
				pickle.dump(le, output)
		else:
			descriptor_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/classes.npy'
			output = open(descriptor_filepath, 'wb')
			pickle.dump(le, output)
	else:
		le = LabelEncoder()
		if config.initialConfig.experiment_folder == None:
			experiment_folder_path = config.initialConfig.root_path + 'experiment_folder_*'
			list_of_folders = glob.glob(experiment_folder_path)
			if list_of_folders == []:
				print("ERROR! No experiment folder found inside the root folder")
				return
			else:
				latest_experiment_folder = max(list_of_folders, key=os.path.getctime)
				descriptor_filepath = latest_experiment_folder + '/classes.pkl'
				pkl_input = open(descriptor_filepath, 'rb')
				le = pickle.load(pkl_input)
		else:
			descriptor_filepath = config.initialConfig.root_path + config.initialConfig.experiment_folder + '/classes.npy'
			pkl_input = open(descriptor_filepath, 'rb')
			le = pickle.load(pkl_input)			
			
	# map each poi id to its respective decoded label
	for key in id_dict:
		id_dict[key][0] = le.transform([id_dict[key][0]])
	
	if args['step'] < 4:
		return le, id_dict, le.transform(labels_set)
	else:
		return le, id_dict, get_class_codes_set_csv(args, None)
	
def get_poi_id_to_class_code_coordinates_dict_csv(conn, args):
	
	"""
	This function returns a dictionary with poi ids as its keys and a 
	list in the form of [< poi's class code >, < x coordinate > < y coordinate >]
	as its values.
	
	Arguments
	---------
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	
	Returns
	-------
	df: :obj:`pandas dataframe`
		the original dataframe with altered column name for the class labels
	poi_id_to_class_code_coordinates_dict: :obj:`dictionary`
		self-explanatory (look at function description)
	"""
	
	df = pd.read_csv(args['pois_csv_name'])
	if args['level'] == 1:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[0]: 'class_code'})
	elif args['level'] == 2:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[1]: 'class_code'})
	else:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[2]: 'class_code'})
	poi_id_to_class_code_coordinates_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	
	for index, row in df.iterrows():
		poi_id_to_class_code_coordinates_dict[row[config.initialConfig.poi_id]] = [row['class_code'], float(row[config.initialConfig.x]), float(row[config.initialConfig.y])]
		
	return df, poi_id_to_class_code_coordinates_dict
	
def get_poi_id_to_boolean_and_counts_per_class_dict_csv(ids, conn, num_of_labels, poi_id_to_encoded_labels_dict, threshold, args):
	
	"""
	This function is responsible for mapping the pois to two lists.
	The first list will contain a  boolean value referring
	to whether a poi of that index's label is within threshold distance
	of the poi whose id is the key of this list in the dictionary. The second list
	contains the respective count of the pois belonging to the
	specific index's label that are within threshold distance of the poi-key.
	
	For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	are within threshold distance of the poi with id = 1, then the dictionaries will look like this: 
	poi_id_to_label_boolean_dict[1] = [1, 0, 1],
	poi_id_to_label_counts_dict[1] = [2, 0, 3]
	
	Arguments
	---------
	ids: :obj:`list`
		a list containing the ids of the pois for which the feature extraction process is to be
		executed
	conn: (redundant)
	num_of_labels: :obj:`int`
		the total number of the different labels
	poi_id_to_encoded_labels_dict: :obj:`dictionary`
		the dictionary mapping the poi ids to labels
	threshold: :obj:`int`
		the aforementioned threshold
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	
	Returns
	-------
	poi_id_to_label_boolean_dict: :obj:`dictionary`
	poi_id_to_label_counts_dict: :obj:`dictionary`
	"""
	
	from sklearn.neighbors import DistanceMetric
	from scipy.spatial import distance
	
	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')
	
	df = pd.read_csv(args['pois_csv_name'])
	
	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
	
	poi_id_to_label_boolean_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	poi_id_to_label_counts_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	
	# add dummy values to the dictionary in order to initialize it
	# in a form that resembles its desired final form
	for poi_id in poi_id_to_label_boolean_dict:
		poi_id_to_label_boolean_dict[poi_id] = [0 for _ in range(0, num_of_labels)]
		
	for poi_id in poi_id_to_label_counts_dict:
		poi_id_to_label_counts_dict[poi_id] = [0 for _ in range(0, num_of_labels)]
	
	
	x = np.asarray(df[config.initialConfig.x])
	y = np.asarray(df[config.initialConfig.y])
	poi_coordinate_array = np.stack((x, y), axis = -1)
	spatial_index = spatial.KDTree(poi_coordinate_array)
	
	for index, row in df.iterrows():
		pois_within_radius = spatial_index.query_ball_point((row[config.initialConfig.x], row[config.initialConfig.y]), config.initialConfig.threshold_distance_neighbor_pois_roads)
		index = int(index)
		if index in pois_within_radius:
			pois_within_radius.remove(index)

		if pois_within_radius is not None:
			for poi_index in pois_within_radius:
				poi_id_to_label_boolean_dict[row[config.initialConfig.poi_id]][poi_id_to_encoded_labels_dict[df.iloc[poi_index][config.initialConfig.poi_id]][0][0]] = 1
				poi_id_to_label_counts_dict[row[config.initialConfig.poi_id]][poi_id_to_encoded_labels_dict[df.iloc[poi_index][config.initialConfig.poi_id]][0][0]] += 1
	"""
	for index1, row1 in df.iterrows():
		for index2, row2 in df.iterrows():
			if row1[config.initialConfig.poi_id] != row2[config.initialConfig.poi_id]:
				# get their coordinates
				point1 = (row1[config.initialConfig.x], row1[config.initialConfig.y])
				point2 = (row2[config.initialConfig.x], row2[config.initialConfig.y])
				# if the two points are within treshold distance, 
				# update the dictionary accordingly
				if distance.euclidean(point1, point2) < config.initialConfig.threshold_distance_neighbor_pois:
					poi_id_to_label_boolean_counts_dict[row1[config.initialConfig.poi_id]][poi_id_to_encoded_labels_dict[row2[config.initialConfig.poi_id]][0][0]][0] = 1
					poi_id_to_label_boolean_counts_dict[row1[config.initialConfig.poi_id]][poi_id_to_encoded_labels_dict[row2[config.initialConfig.poi_id]][0][0]][1] += 1
	"""
	return poi_id_to_label_boolean_dict, poi_id_to_label_counts_dict 
	
def get_closest_pois_boolean_and_counts_per_label_csv(ids, conn, args, threshold = 1000.0):
	
	"""
	A function that just unites the functionality of all previous functions regarding
	the radius (non-street) variant feature extraction.
	
	Arguments
	---------
	ids: :obj:`list`
		the poi ids for which this feature extraction step is to be executed
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	threshold: (redundant)
	
	Returns
	-------
	self-explanatory (refer to the individual functions this function calls)
	"""
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict_csv(conn, args)
	
	# we read the different labels
	class_codes_set = get_class_codes_set_csv(args, df)
	
	# we encode them so we can have a more compact representation of them
	_, poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict_csv(args, class_codes_set, poi_id_to_class_code_coordinates_dict)
	
	return get_poi_id_to_boolean_and_counts_per_class_dict_csv(ids, conn, len(encoded_labels_set), poi_id_to_encoded_labels_dict, threshold, args)

def get_neighbor_pois_boolean_and_counts_per_label_csv(ids, conn, args, threshold = 1000.0):
	
	"""
	A function that just unites the functionality of all previous functions regarding
	the k nearest neighbor variant feature extraction.
	
	Arguments
	---------
	ids: :obj:`list`
		the poi ids for which this feature extraction step is to be executed
	conn: (redundant)
	args: :obj:`dictionary`
		several arguments that are needed for functionality purposes
	threshold: (redundant)
	
	Returns
	-------
	self-explanatory (refer to the individual functions this function calls)
	"""
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict_csv(conn, args)
	
	# we read the different labels
	class_codes_set = get_class_codes_set_csv(args, df)
	
	# we encode them so we can have a more compact representation of them
	_, poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict_csv(args, class_codes_set, poi_id_to_class_code_coordinates_dict)
	
	return get_poi_id_to_neighbors_boolean_and_counts_per_class_dict_csv(ids, conn, len(encoded_labels_set), poi_id_to_encoded_labels_dict, threshold, args)

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-pois_tbl_name", "--pois_tbl_name", required=True,
		help="name of table containing pois information")
	ap.add_argument("-roads_tbl_name", "--roads_tbl_name", required=True,
		help="name of table containing roads information")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()
	
	threshold = 1000.0
	closest_pois_boolean_and_counts_per_label = get_closest_pois_boolean_and_counts_per_label(conn, args, threshold)
	
	closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(conn, args, threshold)
	
if __name__ == "__main__":
   main()
