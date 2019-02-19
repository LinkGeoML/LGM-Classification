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

def get_poi_id_to_closest_poi_ids_dict_csv(ids, conn, args):
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
	
	#is_in_ids = df['poi_id'] in ids
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
	
	for index1, row1 in df.iterrows():
		for index2, row2 in df.iterrows():
			if row1[config.initialConfig.poi_id] != row2[config.initialConfig.poi_id]:
				# get their coordinates
				point1 = (row1[config.initialConfig.x], row1[config.initialConfig.y])
				point2 = (row2[config.initialConfig.x], row2[config.initialConfig.y])
				# if the two points are within treshold distance, 
				# update the dictionary accordingly
				if distance.euclidean(point1, point2) < config.initialConfig.threshold_distance_neighbor_pois_roads:
					#print(distance.euclidean(point1, point2))
					poi_id_to_closet_poi_ids_dict[row1[config.initialConfig.poi_id]].append(row2[config.initialConfig.poi_id])
	
	return poi_id_to_closet_poi_ids_dict
	
def get_poi_id_to_closest_street_id_dict_csv(ids, conn, args):

	"""
	*** This function maps each poi to its closest road id.
	***
	*** Returns - a dictionary consisting of the poi ids as
	*** 		  its keys and their corresponding closest 
	***			  road id as its value.
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
	
	#is_in_ids = poi_df['poi_id'] in ids
	is_in_ids = []
	all_ids = poi_df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	poi_df = poi_df[is_in_ids]
	
	"""
	for index, row in poi_df.iterrows():
		point = Point(row['x'], row['y'])
		print(point)
		point = transform(project, point)
		row['geom'] = point
		print(row['geom'])
	"""
		
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
		
	for index, row in poi_df.iterrows():
		min_distance = 1000000000.0
		poi_geom = Point(row[config.initialConfig.x], row[config.initialConfig.y])
		poi_geom = transform(project, poi_geom)
		#print(point)
		#poi_geom = shapely.wkt.loads(Point(row['x'], row['y']))
		#print(poi_geom)
		#poi_point = (poi_geom.x, poi_geom.y)
		for index1, row1 in street_df.iterrows():
			street_geom = row1['geometry']
			#print(row1['geometry'])
			#print(row1['geometry'].distance(poi_geom))
			#street_geom = row1['geometry']
			# get the index of the minimum one and map the corresponding edge id to the poi
			#street_x = street_geom.x
			#street_y = street_geom.y
			#street_point = (street_geom.x, street_geom.y)
			#poi_id_to_edge_id_dict[row['poi_id']][1] = street_geom
			if street_geom.distance(poi_geom) < min_distance:
				min_distance = street_geom.distance(poi_geom)
				#print(min_distance)
				poi_id_to_street_geom_dict[row[config.initialConfig.poi_id]] = street_geom

	return poi_id_to_street_geom_dict, street_df
	
def get_street_geom_to_closest_poi_ids_dict_csv(ids, conn, args, street_df):
	
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
	
	#is_in_ids = poi_df['poi_id'] in ids
	is_in_ids = []
	all_ids = poi_df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	poi_df = poi_df[is_in_ids]
	
	for index, row in poi_df.iterrows():
		point = Point(row[config.initialConfig.x], row[config.initialConfig.y])
		point = transform(project, point)
		row['geom'] = point

	geom_keys = []
	for index, row in street_df.iterrows():
		geom_keys.append(str(row['geometry']))
	#print(geom_keys)
	street_geom_to_to_closest_poi_ids_dict = dict.fromkeys(geom_keys)
	#print(street_geom_to_to_closest_poi_ids_dict)
	for street_geom in street_geom_to_to_closest_poi_ids_dict:
		street_geom_to_to_closest_poi_ids_dict[street_geom] = []
	
	for index, row in street_df.iterrows():
		street_geom = row['geometry']
		#street_point = (street_geom.x, street_geom.y)
		for index1, row1 in poi_df.iterrows():
			poi_geom = shapely.wkt.loads(row1['geom'])
			#poi_point = (poi_geom.x, poi_geom.y)
			if street_geom.distance(poi_geom) < config.initialConfig.threshold_distance_neighbor_pois_roads:
				#print(street_geom.distance(poi_geom) * 111000)
				street_geom_to_to_closest_poi_ids_dict[str(street_geom)].append([row1[config.initialConfig.poi_id], row1['class_code']])
				
	return street_geom_to_to_closest_poi_ids_dict
	
def construct_final_feature_vector_csv(ids, conn, args, num_of_labels, poi_id_to_encoded_labels_dict, poi_id_to_closest_poi_ids_dict, poi_id_to_closest_street_id_dict, street_geom_to_closest_poi_ids_dict):
	
	poi_df = pd.read_csv(args['pois_csv_name'])
	if args['level'] == 1:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[0]: 'class_code'})
	elif args['level'] == 2:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[1]: 'class_code'})
	else:
		poi_df = poi_df.rename(index = str, columns = {config.initialConfig.class_codes[2]: 'class_code'})
	
	#is_in_ids = df['poi_id'] in ids
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
	
	poi_id_to_closest_pois_street_boolean_and_counts_per_label_dict = dict.fromkeys(poi_id_to_closest_poi_ids_dict.keys())
	
	# get the class codes set and encode the class codes to labels
	#class_codes_set = get_class_codes_set(args, poi_df)
	#id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict(class_codes_set, id_dict)
	#num_of_labels = len(encoded_labels_set)
	#print(id_to_encoded_labels_dict)
	
	# prepare the street id dictionary to be able to store the
	# boolean and count duplet for each of the class labels
	for poi_id in poi_id_to_closest_pois_street_boolean_and_counts_per_label_dict:
		poi_id_to_closest_pois_street_boolean_and_counts_per_label_dict[poi_id] = [[0,0] for _ in range(0, num_of_labels)]
		
	#print(num_of_labels)
	
	for poi_id in poi_id_to_closest_street_id_dict:
		street_geometry = poi_id_to_closest_street_id_dict[poi_id]
		#print(street_geom_to_closest_poi_ids_dict[str(street_geometry)])
		for street_neighboring_poi_id_class_code_list in street_geom_to_closest_poi_ids_dict[str(street_geometry)]:
			# uncomment the following line in order to check radius from original poi
			#if street_neighboring_poi_id_class_code_list[0] in poi_id_to_closest_poi_ids_dict:
				#print(street_neighboring_poi_id_class_code_list[0])
				#print(id_to_encoded_labels_dict[street_neighboring_poi_id_class_code_list[0]])
				#print(id_to_encoded_labels_dict[street_neighboring_poi_id_class_code_list[0]][0][0])
				poi_id_to_closest_pois_street_boolean_and_counts_per_label_dict[poi_id][poi_id_to_encoded_labels_dict[street_neighboring_poi_id_class_code_list[0]][0][0]][0] = 1
				poi_id_to_closest_pois_street_boolean_and_counts_per_label_dict[poi_id][poi_id_to_encoded_labels_dict[street_neighboring_poi_id_class_code_list[0]][0][0]][1] += 1
				
	return poi_id_to_closest_pois_street_boolean_and_counts_per_label_dict
	
def get_closest_pois_boolean_and_counts_per_label_streets_csv(ids, conn, args, threshold = 1000.0):
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict_csv(conn, args)
	
	# we read the different labels
	class_codes_set = get_class_codes_set_csv(args, df)
	
	# we encode them so we can have a more compact representation of them
	poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict_csv(class_codes_set, poi_id_to_class_code_coordinates_dict)
	
	poi_id_to_closest_poi_ids_dict = get_poi_id_to_closest_poi_ids_dict_csv(ids, conn, args)
	# get the dictionary mapping each poi id to that of its closest road
	poi_id_to_closest_street_id_dict, street_df = get_poi_id_to_closest_street_id_dict_csv(ids, conn, args)
	
	street_geom_to_closest_poi_ids_dict = get_street_geom_to_closest_poi_ids_dict_csv(ids, conn, args, street_df)
	
	final_feature_vector = construct_final_feature_vector_csv(ids, conn, args, len(encoded_labels_set), poi_id_to_encoded_labels_dict, poi_id_to_closest_poi_ids_dict, poi_id_to_closest_street_id_dict, street_geom_to_closest_poi_ids_dict)
	
	#print(final_feature_vector)
	return final_feature_vector
	
def get_class_codes_set_csv(args, df):
	
	"""
	*** This function is responsible for reading the excel file
	*** containing the dataset labels (here stored in a more code-like
	*** manner rather than resembling labels).
	***
	*** Returns - a list of the class codes
	"""
	
	# read the file containing the class codes
	#df = pd.read_excel('/home/nikos/Desktop/Datasets/GeoData_PoiMarousi/GeoData_poiClasses.xlsx', sheet_name=None)		
	
	# store the class codes (labels) in the list
	"""
	if config.initialConfig.level == 1:
		class_codes = list(df['theme'])
		#print(class_codes)
	elif config.initialConfig.level == 2:
		class_codes = list(df['class_name'])
	else:
		class_codes = list(df['subclass_n'].dropna())
		#print(class_codes)
	"""
	class_codes = list(df['class_code'])
	#print(np.unique(class_codes))
	class_codes = list(set(class_codes))
	return class_codes
	
def get_poi_id_to_encoded_labels_dict_csv(labels_set, id_dict):
	
	"""
	*** This function encodes our labels to values between 0 and len(labels_set)
	*** in order to have a more compact and user-friendly encoding of them.
	***
	*** Arguments - labels_set: the set of the labels (class codes) as we
	*** 			extracted them from the excel file
	***				id_dict: the dictionary containing the ids of the pois
	***
	*** Returns -	id_dict: an updated version of our pois dictionary
	***						 now mapping their ids to their encoded labels
	***				labels_set: the encoded labels set
	"""
	
	from sklearn.preprocessing import LabelEncoder
	
	# fit the label encoder to our labels set
	le = LabelEncoder()
	le.fit(labels_set)
	
	# map each poi id to its respective decoded label
	for key in id_dict:
		id_dict[key][0] = le.transform([id_dict[key][0]])
	
	return id_dict, le.transform(labels_set)
	
def get_poi_id_to_class_code_coordinates_dict_csv(conn, args):
	
	"""
	*** This function returns a dictionary with poi ids as its keys and a 
	*** list in the form of [< poi's class code >, < x coordinate > < y coordinate >]
	*** as its values.
	"""
	
	df = pd.read_csv(args['pois_csv_name'])
	#print(df.columns)
	if args['level'] == 1:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[0]: 'class_code'})
	elif args['level'] == 2:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[1]: 'class_code'})
	else:
		df = df.rename(index = str, columns = {config.initialConfig.class_codes[2]: 'class_code'})
	#print(df.columns)
	poi_id_to_class_code_coordinates_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	
	for index, row in df.iterrows():
		poi_id_to_class_code_coordinates_dict[row[config.initialConfig.poi_id]] = [row['class_code'], float(row[config.initialConfig.x]), float(row[config.initialConfig.y])]
		"""
		if config.initialConfig.level == 1:
			poi_id_to_class_code_coordinates_dict[row['id']] = [row['theme'], float(row['x']), float(row['y'])]
		elif config.initialConfig.level == 2:
			poi_id_to_class_code_coordinates_dict[row['id']] = [row['class_name'], float(row['x']), float(row['y'])]
		else:
			poi_id_to_class_code_coordinates_dict[row['id']] = [row['subclass_n'], float(row['x']), float(row['y'])]
		"""
	return df, poi_id_to_class_code_coordinates_dict
	
def get_poi_id_to_boolean_and_counts_per_class_dict_csv(ids, conn, num_of_labels, poi_id_to_encoded_labels_dict, threshold, args):
	
	"""
	*** This function is responsible for mapping the pois to a list of two-element lists.
	*** The first element of that list will contain a  boolean value referring
	*** to whether a poi of that index's label is within threshold distance
	*** of the poi whose id is the key of this list in the dictionary. The second
	*** element contains the respective count of the pois belonging to the
	*** specific index's label that are within threshold distance of the poi-key.
	***
	*** For example, if two pois, zero pois and three pois from classes 0, 1 and 2 respectively
	*** are within threshold distance of the poi with id = 1, then the dictionary will look like this: 
	*** id_dict[1] = [[1, 2], [0, 0], [1, 3]]
	***
	*** Arguments - num_of_labels: the total number of the different labels
	*** 			encoded_labels_id_dict: the dictionary mapping the poi ids to labels
	***				threshold: the aforementioned threshold
	"""
	
	from sklearn.neighbors import DistanceMetric
	from scipy.spatial import distance
	
	# define the apropriate distance metric for measuring distance between two pois
	dist = DistanceMetric.get_metric('euclidean')
	
	df = pd.read_csv(args['pois_csv_name'])
	
	poi_id_to_label_boolean_counts_dict = dict.fromkeys(df[config.initialConfig.poi_id])
	
	#is_in_ids = df['poi_id'] in ids
	is_in_ids = []
	all_ids = df[config.initialConfig.poi_id]
	for id in all_ids:
		if id in ids:
			is_in_ids.append(True)
		else:
			is_in_ids.append(False)
	df = df[is_in_ids]
	
	# add dummy values to the dictionary in order to initialize it
	# in a form that resembles its desired final form
	for poi_id in poi_id_to_label_boolean_counts_dict:
		poi_id_to_label_boolean_counts_dict[poi_id] = [[0,0] for _ in range(0, num_of_labels)]
	
	#print(num_of_labels)
	
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
	
	#print(poi_id_to_label_boolean_counts_dict)
	return poi_id_to_label_boolean_counts_dict
	
def get_closest_pois_boolean_and_counts_per_label_csv(ids, conn, args, threshold = 1000.0):
	
	"""
	*** This function returns a dictionary with the poi ids as its keys
	*** and two lists for each key. The first list contains boolean values
	*** dictating whether a poi of that index's label is within threshold
	*** distance with the key poi. The second list contains the counts of
	*** the pois belonging to the same index's label.
	
	*** Arguments - threshold: we only examine pois the distance between 
	*** 			which is below the given threshold
	"""
	
	# we build a dictionary containing the poi ids as keys
	# and we map to it its x, y coordinates
	df, poi_id_to_class_code_coordinates_dict = get_poi_id_to_class_code_coordinates_dict_csv(conn, args)
	
	# we read the different labels
	class_codes_set = get_class_codes_set_csv(args, df)
	
	# we encode them so we can have a more compact representation of them
	poi_id_to_encoded_labels_dict, encoded_labels_set = get_poi_id_to_encoded_labels_dict_csv(class_codes_set, poi_id_to_class_code_coordinates_dict)
	
	return get_poi_id_to_boolean_and_counts_per_class_dict_csv(ids, conn, len(encoded_labels_set), poi_id_to_encoded_labels_dict, threshold, args)

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
	#print(closest_pois_boolean_and_counts_per_label)
	
	closest_pois_boolean_and_counts_per_label_streets = get_closest_pois_boolean_and_counts_per_label_streets(conn, args, threshold)
	#print(closest_pois_boolean_and_counts_per_label_streets)
	
if __name__ == "__main__":
   main()
