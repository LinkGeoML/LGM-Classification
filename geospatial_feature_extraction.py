#!/usr/bin/python

# import the necessary packages
import geopandas as gpd
import psycopg2
import argparse
import numpy as np
from database import *
from preprocessing import *

def get_area_mxm(conn, args):
	
	# this function returns the area (in sq. metres) of the rows of a given table

	sql = "select id, ST_Area(geom) as mxm, geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	id_dictionary = dict.fromkeys(df['id'])
	
	for index, row in df.iterrows():
		id_dictionary[row['id']] = row['mxm']

	return id_dictionary
	
def get_perimeter(conn, args):
	
	# this function returns the perimeter of the rows of a given table
		
	sql = "select id, ST_Perimeter(geom) as perimeter, geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	id_dictionary = dict.fromkeys(df['id'])
	
	for index, row in df.iterrows():
		id_dictionary[row['id']] = row['perimeter']

	return id_dictionary
	
def get_vertices(conn, args):
	
	# this function returns the number of vertices of the rows of a given table

	sql = "select id, sum(ST_NPoints(geom)) as num_of_vertices, geom from {0} group by id".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	id_dictionary = dict.fromkeys(df['id'])
	
	for index, row in df.iterrows():
		id_dictionary[row['id']] = row['num_of_vertices']
		
	return id_dictionary
	
def get_touches(conn, args):
	
	# returns a dictionary which contains as keys the id of each geometry object
	# and holds two values: a boolean value stating whether this particular object
	# touches with another object and a numeric one referint to how many distinct objects it touches
	
	sql = "select id, geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	id_dictionary = dict.fromkeys(df['id'])
	for key in id_dictionary:
		id_dictionary[key] = [0, 0]
		
	for index, row in df.iterrows():
				
		sql_touches = "SELECT * FROM {0} WHERE ST_Touches({0}.geom,(SELECT {0}.geom FROM {0} WHERE {0}.id = {1}))".format(args["pois_tbl_name"], row['id'])
		df_touches = gpd.GeoDataFrame.from_postgis(sql_touches, conn)
		for index, row in df_touches.iterrows():
			id_dictionary[row['id']][0] = 1
			id_dictionary[row['id']][1] += 1
			
	return id_dictionary
	
def get_intersects(conn, args):
	
	# returns a dictionary which contains as keys the id of each geometry object
	# and holds two values: a boolean value stating whether this particular object
	# touches with another object and a numeric one referint to how many distinct objects it touches
	
	sql = "select id, geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	id_dictionary = dict.fromkeys(df['id'])
	for key in id_dictionary:
		id_dictionary[key] = [0, 0]
		
	for index, row in df.iterrows():
				
		sql_touches = "SELECT * FROM {0} WHERE ST_Intersects({0}.geom,(SELECT {0}.geom FROM {0} WHERE {0}.id = {1}))".format(args["pois_tbl_name"], row['id'])
		df_touches = gpd.GeoDataFrame.from_postgis(sql_touches, conn)
		for index, row in df_touches.iterrows():
			id_dictionary[row['id']][0] = 1
			id_dictionary[row['id']][1] += 1
			
	return id_dictionary
	
def get_covers(conn, args):
	
	# returns a dictionary which contains as keys the id of each geometry object
	# and holds two values: a boolean value stating whether this particular object
	# touches with another object and a numeric one referint to how many distinct objects it touches
	
	sql = "select id, geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	id_dictionary = dict.fromkeys(df['id'])
	for key in id_dictionary:
		id_dictionary[key] = [0, 0]
		
	for index, row in df.iterrows():
				
		sql_touches = "SELECT * FROM {0} WHERE ST_Covers({0}.geom,(SELECT {0}.geom FROM {0} WHERE {0}.id = {1}))".format(args["pois_tbl_name"], row['id'])
		df_touches = gpd.GeoDataFrame.from_postgis(sql_touches, conn)
		for index, row in df_touches.iterrows():
			id_dictionary[row['id']][0] = 1
			id_dictionary[row['id']][1] += 1
			
	return id_dictionary
	
def get_coveredbys(conn, args):
	
	# returns a dictionary which contains as keys the id of each geometry object
	# and holds two values: a boolean value stating whether this particular object
	# touches with another object and a numeric one referint to how many distinct objects it touches
	
	sql = "select id, geom from {0}".format(args["pois_tbl_name"])
	df = gpd.GeoDataFrame.from_postgis(sql, conn, geom_col = 'geom')
	
	id_dictionary = dict.fromkeys(df['id'])
	for key in id_dictionary:
		id_dictionary[key] = [0, 0]
		
	for index, row in df.iterrows():
				
		sql_touches = "SELECT * FROM {0} WHERE ST_CoveredBy({0}.geom,(SELECT {0}.geom FROM {0} WHERE {0}.id = {1}))".format(args["pois_tbl_name"], row['id'])
		df_touches = gpd.GeoDataFrame.from_postgis(sql_touches, conn)
		for index, row in df_touches.iterrows():
			id_dictionary[row['id']][0] = 1
			id_dictionary[row['id']][1] += 1
			
	return id_dictionary
	
def get_statistics_for_edges(conn, args):
	
	# this function returns the mean and variance of the number of edges of the rows of a given table

	vertices = get_vertices(conn, args)
	vertices = np.asarray(vertices)
	
	return np.mean(vertices), np.var(vertices)
	
def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-tbl_name", "--tbl_name", required=True,
		help="table name")
	args = vars(ap.parse_args())
	
	# call the appropriate function to connect to the database
	conn = connect_to_db()

	# Trying GIS queries
	
	
	areas = get_area_mxm(conn, args)
	areas = np.asarray(areas)
	
	perimeters = get_perimeter(conn, args)
	perimeters = np.asarray(perimeters)
	
	vertices = get_perimeter(conn, args)
	vertices = np.asarray(vertices_list)
	
	mean_edges, variance_edges = get_statistics_for_edges(conn, args)
	mean_edges = np.asarray(mean_edges)
	variance_edges = np.asarray(variance_edges)
	
	touches = get_touches(conn, args)
	
	#X_train, X_test = standardize_data()

if __name__ == "__main__":
   main()
