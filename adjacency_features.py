import numpy as np
import pickle

from sklearn.neighbors import KDTree


def create_poi_index(poi_gdf, path):
    poi_coords = poi_gdf[['lon', 'lat']].values
    poi_index = KDTree(poi_coords)
    pickle.dump(poi_index, open(path, 'wb'))
    return


def get_classes_in_radius_bln(poi_gdf, poi_index_path, nlabels, label_map, thr):
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
    street_index = street_gdf.sindex
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = (poi.lon, poi.lat)
        result_street_idx = list(street_index.nearest(poi_coords))[0]
        result_pois_idxs = [
            poi_idx for poi_idx in pois_by_street[result_street_idx]
            if poi.geometry.distance(geometry_map[poi_idx]) < thr]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            X[poi.Index][rpi_label] = 1
    return X


def get_classes_in_street_and_radius_cnt(poi_gdf, street_gdf, pois_by_street, nlabels, label_map, geometry_map, thr):
    street_index = street_gdf.sindex
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = (poi.lon, poi.lat)
        result_street_idx = list(street_index.nearest(poi_coords))[0]
        class_cnt = dict((c, 0) for c in range(nlabels))
        result_pois_idxs = [
            poi_idx for poi_idx in pois_by_street[result_street_idx]
            if poi.geometry.distance(geometry_map[poi_idx]) < thr]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            class_cnt[rpi_label] += 1
        for k, v in class_cnt.items():
            X[poi.Index][k] = v
    return X


def get_classes_in_neighbors_bln(poi_gdf, poi_index_path, nlabels, label_map, k):
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
        result_street_idx = list(street_index.nearest(poi_coords))[0]
        result_pois_idxs = [
            i for i, geom in enumerate(geometry_map)
            if street_gdf.iloc[result_street_idx]['geometry'].distance(geom) < thr]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            X[poi.Index][rpi_label] = 1
    return X


def get_classes_in_street_radius_cnt(poi_gdf, street_gdf, nlabels, label_map, geometry_map, thr):
    street_index = street_gdf.sindex
    X = np.zeros((len(poi_gdf), nlabels))
    for poi in poi_gdf.itertuples():
        poi_coords = (poi.lon, poi.lat)
        result_street_idx = list(street_index.nearest(poi_coords))[0]
        class_cnt = dict((c, 0) for c in range(nlabels))
        result_pois_idxs = [
            i for i, geom in enumerate(geometry_map)
            if street_gdf.iloc[result_street_idx]['geometry'].distance(geom) < thr]
        for rpi in result_pois_idxs:
            rpi_label = label_map[rpi]
            class_cnt[rpi_label] += 1
        for k, v in class_cnt.items():
            X[poi.Index][k] = v
    return X
