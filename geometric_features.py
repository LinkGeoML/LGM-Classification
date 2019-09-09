import numpy as np


features_getter_map = {
    'area': 'get_area',
    'perimeter': 'get_perimeter',
    'n_vertices': 'get_n_vertices',
    'mean_edge_length': 'get_mean_edge_length',
    'var_edge_length': 'get_var_edge_length'
}


def get_area(poi_gdf):
    X = np.zeros((len(poi_gdf), 1))
    return X


def get_perimeter(poi_gdf):
    X = np.zeros((len(poi_gdf), 1))
    return X


def get_n_vertices(poi_gdf):
    X = np.zeros((len(poi_gdf), 1))
    return X


def get_mean_edge_length(poi_gdf):
    X = np.zeros((len(poi_gdf), 1))
    return X


def get_var_edge_length(poi_gdf):
    X = np.zeros((len(poi_gdf), 1))
    return X
