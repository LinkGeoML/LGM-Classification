import numpy as np
from shapely.geometry import Point
import pickle

from sklearn.neighbors import KDTree


def create_poi_index(poi_gdf, path):
    """
    Creates spatial index containing the pois given.

    Args:
        poi_gdf (geopandas.GeoDataFrame) : Contains pois to be stored in the \
            index
        path (str): Path to save the index

    Returns:
        None
    """
    poi_coords = poi_gdf[['lon', 'lat']].values
    poi_index = KDTree(poi_coords)
    pickle.dump(poi_index, open(path, 'wb'))
    return


def get_classes_in_radius_bln(poi_gdf, poi_index_path, nlabels, label_map, thr):
    """
    Creates a features array. For each poi *p* (each row) the array will
    contain 1 (True) in column *c*, if there is at least one poi of category *c*
    inside *p*'s defined radius.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        poi_index_path (str): Path to the stored index
        nlabels (int): Number of poi categories
        label_map (list): A list containing the labels of the train pois
        thr (float): Radius to be searched (in meters)

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), nlabels)
    """
    poi_index = pickle.load(open(poi_index_path, 'rb'))
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = np.array([poi.lon, poi.lat]).reshape(1, -1)
        result_pois_idxs = poi_index.query_radius(poi_coords, r=thr)[0]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            X[poi.Index][rpi_label] = 1
    return X


def get_classes_in_radius_cnt(poi_gdf, poi_index_path, nlabels, label_map, thr):
    """
    Creates a features array. For each poi *p* (each row) the array will
    contain an integer in column *c*, representing the number of pois of
    category *c* inside *p*'s defined radius.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        poi_index_path (str): Path to the stored index
        nlabels (int): Number of poi categories
        label_map (list): A list containing the labels of the train pois
        thr (float): Radius to be searched (in meters)

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), nlabels)
    """
    poi_index = pickle.load(open(poi_index_path, 'rb'))
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        class_cnt = dict((c, 0) for c in range(nlabels))
        poi_coords = np.array([poi.lon, poi.lat]).reshape(1, -1)
        result_pois_idxs = poi_index.query_radius(poi_coords, r=thr)[0]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            class_cnt[rpi_label] += 1
        for k, v in class_cnt.items():
            X[poi.Index][k] = v
    return X


def get_classes_in_street_and_radius_bln(poi_gdf, street_gdf, pois_by_street, nlabels, label_map, geometry_map, thr):
    """
    Creates a features array. For each poi *p*, the nearest street to *p* is \
    identified and the pois of this street are kept. These pois are then \
    filtered and only those which are inside *p*'s defined radius are \
    considered (e.g. a set of pois *P*). Finally, for each poi *p* (each row) \
    the array will contain 1 (True) in column *c*, if there is at least one \
    poi of category *c* among pois in *P*.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        street_gdf (geopandas.GeoDataFrame): Contains all streets extracted \
            from OSM, along with their geometries
        pois_by_street (dict): Has streets ids as keys and a list containing \
            the pois which belong to each street as values
        nlabels (int): Number of poi categories
        label_map (list): A list containing the labels of the train pois
        geometry_map (list): A list containing the geometries of the train pois
        thr (float): Radius to be searched (in meters)

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), nlabels)
    """
    street_index = street_gdf.sindex
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = (poi.lon, poi.lat)
        candidates = list(street_index.nearest(poi_coords))
        nearest = candidates[np.argmin([
            Point(poi_coords).distance(street_gdf.iloc[c]['geometry'])
            for c in candidates
        ])]
        result_pois_idxs = [
            poi_idx for poi_idx in pois_by_street[nearest]
            if poi.geometry.distance(geometry_map[poi_idx]) < thr]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            X[poi.Index][rpi_label] = 1
    return X


def get_classes_in_street_and_radius_cnt(poi_gdf, street_gdf, pois_by_street, nlabels, label_map, geometry_map, thr):
    """
    Creates a features array. For each poi *p*, the nearest street to *p* is \
    identified and the pois of this street are kept. These pois are then \
    filtered and only those which are inside *p*'s defined radius are \
    considered (e.g. a set of pois *P*). Finally, for each poi *p* (each row) \
    the array will contain an integer in column *c*, representing the number \
    of pois of category *c* among pois in *P*.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        street_gdf (geopandas.GeoDataFrame): Contains all streets extracted \
            from OSM, along with their geometries
        pois_by_street (dict): Has streets ids as keys and a list containing \
            the pois which belong to each street as values
        nlabels (int): Number of poi categories
        label_map (list): A list containing the labels of the train pois
        geometry_map (list): A list containing the geometries of the train pois
        thr (float): Radius to be searched (in meters)

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), nlabels)
    """
    street_index = street_gdf.sindex
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = (poi.lon, poi.lat)
        candidates = list(street_index.nearest(poi_coords))
        nearest = candidates[np.argmin([
            Point(poi_coords).distance(street_gdf.iloc[c]['geometry'])
            for c in candidates
        ])]
        class_cnt = dict((c, 0) for c in range(nlabels))
        result_pois_idxs = [
            poi_idx for poi_idx in pois_by_street[nearest]
            if poi.geometry.distance(geometry_map[poi_idx]) < thr]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            class_cnt[rpi_label] += 1
        for k, v in class_cnt.items():
            X[poi.Index][k] = v
    return X


def get_classes_in_neighbors_bln(poi_gdf, poi_index_path, nlabels, label_map, k):
    """
    Creates a features array. For each poi *p* (each row) the array will \
    contain 1 (True) in column *c*, if there is at least one poi of category \
    *c* among the *k* nearest neighbors of *p*.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        poi_index_path (str): Path to the stored index
        nlabels (int): Number of poi categories
        label_map (list): A list containing the labels of the train pois
        k (int): Number of nearest neighbors to take into account

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), nlabels)
    """
    poi_index = pickle.load(open(poi_index_path, 'rb'))
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = np.array([poi.lon, poi.lat]).reshape(1, -1)
        result_pois_idxs = poi_index.query(poi_coords, k=k)[1][0]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            X[poi.Index][rpi_label] = 1
    return X


def get_classes_in_neighbors_cnt(poi_gdf, poi_index_path, nlabels, label_map, k):
    """
    Creates a features array. For each poi *p* (each row) the array will \
    contain an integer in column *c*, representing the number of pois of \
    category *c* among the *k* nearest neighbors of *p*.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        poi_index_path (str): Path to the stored index
        nlabels (int): Number of poi categories
        label_map (list): A list containing the labels of the train pois
        k (int): Number of nearest neighbors to take into account

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), nlabels)
    """
    poi_index = pickle.load(open(poi_index_path, 'rb'))
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        class_cnt = dict((c, 0) for c in range(nlabels))
        poi_coords = np.array([poi.lon, poi.lat]).reshape(1, -1)
        result_pois_idxs = poi_index.query(poi_coords, k=k)[1][0]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            class_cnt[rpi_label] += 1
        for k, v in class_cnt.items():
            X[poi.Index][k] = v
    return X


def get_classes_in_street_radius_bln(poi_gdf, street_gdf, nlabels, label_map, geometry_map, thr):
    street_index = street_gdf.sindex
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = (poi.lon, poi.lat)
        candidates = list(street_index.nearest(poi_coords))
        nearest = candidates[np.argmin([
            Point(poi_coords).distance(street_gdf.iloc[c]['geometry'])
            for c in candidates
        ])]
        result_pois_idxs = [
            i for i, geom in enumerate(geometry_map)
            if street_gdf.iloc[nearest]['geometry'].distance(geom) < thr]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            X[poi.Index][rpi_label] = 1
    return X


def get_classes_in_street_radius_cnt(poi_gdf, street_gdf, nlabels, label_map, geometry_map, thr):
    street_index = street_gdf.sindex
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = (poi.lon, poi.lat)
        candidates = list(street_index.nearest(poi_coords))
        nearest = candidates[np.argmin([
            Point(poi_coords).distance(street_gdf.iloc[c]['geometry'])
            for c in candidates
        ])]
        class_cnt = dict((c, 0) for c in range(nlabels))
        result_pois_idxs = [
            i for i, geom in enumerate(geometry_map)
            if street_gdf.iloc[nearest]['geometry'].distance(geom) < thr]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            class_cnt[rpi_label] += 1
        for k, v in class_cnt.items():
            X[poi.Index][k] = v
    return X
