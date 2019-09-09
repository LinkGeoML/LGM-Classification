import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.wkt import loads
import itertools
import os
from collections import Counter
import pickle

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2


import adjacency_features as af
import textual_features as tf
# import geometric_features as gf
# import matching as m
import osm_utilities as osm_ut
import writers as wrtrs
from config import config


feature_module_map = {
    'classes_in_radius_bln': af,
    'classes_in_radius_cnt': af,
    'classes_in_street_and_radius_bln': af,
    'classes_in_street_and_radius_cnt': af,
    'classes_in_neighbors_bln': af,
    'classes_in_neighbors_cnt': af,
    'classes_in_street_radius_bln': af,
    'classes_in_street_radius_cnt': af,
    'similarity_per_class': tf,
    'top_k_terms': tf,
    'top_k_trigrams': tf,
    'top_k_fourgrams': tf
}

features_getter_map = {
    'classes_in_radius_bln': 'get_classes_in_radius_bln',
    'classes_in_radius_cnt': 'get_classes_in_radius_cnt',
    'classes_in_street_and_radius_bln': 'get_classes_in_street_and_radius_bln',
    'classes_in_street_and_radius_cnt': 'get_classes_in_street_and_radius_cnt',
    'classes_in_neighbors_bln': 'get_classes_in_neighbors_bln',
    'classes_in_neighbors_cnt': 'get_classes_in_neighbors_cnt',
    'classes_in_street_radius_bln': 'get_classes_in_street_radius_bln',
    'classes_in_street_radius_cnt': 'get_classes_in_street_radius_cnt',
    'similarity_per_class': 'get_similarity_per_class',
    'top_k_terms': 'get_top_k_terms',
    'top_k_trigrams': 'get_top_k_trigrams',
    'top_k_fourgrams': 'get_top_k_fourgrams'
}

features_params_map = {
    'classes_in_radius_bln': 'classes_in_radius_thr',
    'classes_in_radius_cnt': 'classes_in_radius_thr',
    'classes_in_street_and_radius_bln': 'classes_in_street_and_radius_thr',
    'classes_in_street_and_radius_cnt': 'classes_in_street_and_radius_thr',
    'classes_in_neighbors_bln': 'classes_in_neighbors_thr',
    'classes_in_neighbors_cnt': 'classes_in_neighbors_thr',
    'classes_in_street_radius_bln': 'classes_in_street_radius_thr',
    'classes_in_street_radius_cnt': 'classes_in_street_radius_thr',
    'top_k_terms': 'top_k_terms_pct',
    'top_k_trigrams': 'top_k_trigrams_pct',
    'top_k_fourgrams': 'top_k_fourgrams_pct'
}

features_getter_args_map = {
    'classes_in_radius_bln': ('poi_gdf', 'poi_index_path', 'nlabels', 'label_map', 'param'),
    'classes_in_radius_cnt': ('poi_gdf', 'poi_index_path', 'nlabels', 'label_map', 'param'),
    'classes_in_street_and_radius_bln': ('poi_gdf', 'street_gdf', 'pois_by_street', 'nlabels', 'label_map', 'geometry_map', 'param'),
    'classes_in_street_and_radius_cnt': ('poi_gdf', 'street_gdf', 'pois_by_street', 'nlabels', 'label_map', 'geometry_map', 'param'),
    'classes_in_neighbors_bln': ('poi_gdf', 'poi_index_path', 'nlabels', 'label_map', 'param'),
    'classes_in_neighbors_cnt': ('poi_gdf', 'poi_index_path', 'nlabels', 'label_map', 'param'),
    'classes_in_street_radius_bln': ('poi_gdf', 'street_gdf', 'nlabels', 'label_map', 'geometry_map', 'param'),
    'classes_in_street_radius_cnt': ('poi_gdf', 'street_gdf', 'nlabels', 'label_map', 'geometry_map', 'param'),
    'similarity_per_class': ('poi_gdf', 'textual_index_path', 'nlabels'),
    'top_k_terms': ('poi_gdf', 'names', 'param'),
    'top_k_trigrams': ('poi_gdf', 'names', 'param'),
    'top_k_fourgrams': ('poi_gdf', 'names', 'param')
}


def load_poi_gdf(poi_fpath):
    poi_df = pd.read_csv(poi_fpath)
    poi_df['geometry'] = poi_df.apply(
        lambda x: Point(x[config.lon_col], x[config.lat_col]), axis=1)
    poi_gdf = gpd.GeoDataFrame(poi_df, geometry='geometry')
    poi_gdf.crs = {'init': f'epsg:{config.poi_crs}'}
    poi_gdf = poi_gdf.to_crs({'init': 'epsg:3857'})
    poi_gdf['lon'] = poi_gdf.apply(lambda p: p.geometry.coords[0][0], axis=1)
    poi_gdf['lat'] = poi_gdf.apply(lambda p: p.geometry.coords[0][1], axis=1)
    return poi_gdf


def encode_labels(poi_gdf, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
        poi_gdf['label'] = encoder.fit_transform(poi_gdf[config.label_col])
    else:
        poi_gdf = poi_gdf[poi_gdf[config.label_col].isin(encoder.classes_)].reset_index(drop=True)
        poi_gdf['label'] = encoder.transform(poi_gdf[config.label_col])
    return poi_gdf, encoder


def load_street_gdf(street_fpath):
    street_df = pd.read_csv(street_fpath)
    street_df['geometry'] = street_df['geometry'].apply(lambda x: loads(x))
    street_gdf = gpd.GeoDataFrame(street_df, geometry='geometry')
    street_gdf.crs = {'init': f'epsg:{config.osm_crs}'}
    street_gdf = street_gdf.to_crs({'init': 'epsg:3857'})
    return street_gdf


def load_poly_gdf(poly_fpath):
    poly_df = pd.read_csv(poly_fpath)
    poly_df['geometry'] = poly_df['geometry'].apply(lambda x: loads(x))
    poly_gdf = gpd.GeoDataFrame(poly_df, geometry='geometry')
    poly_gdf.crs = {'init': f'epsg:{config.osm_crs}'}
    poly_gdf = poly_gdf.to_crs({'init': 'epsg:3857'})
    return poly_gdf


def get_bbox_coords(poi_gdf):
    poi_gdf = poi_gdf.to_crs({'init': f'epsg:{config.osm_crs}'})
    min_lon, min_lat, max_lon, max_lat = poi_gdf.geometry.total_bounds
    return (min_lat, min_lon, max_lat, max_lon)


def get_required_external_files(poi_gdf, feature_sets_path):
    if (
        'classes_in_street_and_radius_bln' in config.included_adjacency_features or
        'classes_in_street_and_radius_cnt' in config.included_adjacency_features or
        'classes_in_street_radius_bln' in config.included_adjacency_features or
        'classes_in_street_radius_cnt' in config.included_adjacency_features
       ):
        osm_ut.download_osm_streets(get_bbox_coords(poi_gdf), feature_sets_path)
    # if config.included_geometric_features:
    #     osm_ut.download_osm_polygons(get_bbox_coords(poi_gdf), feature_sets_path)
    return


def ngrams(n, word):
    for i in range(len(word)-n-1):
        yield word[i:i+n]


def get_top_k(names, k, mode='term'):
    if mode == 'trigram':
        cnt = Counter(ngram for word in names for ngram in ngrams(3, word))
    elif mode == 'fourgram':
        cnt = Counter(ngram for word in names for ngram in ngrams(4, word))
    else:
        cnt = Counter(names)
    return [t[0] for t in cnt.most_common(int(len(cnt) * k))]


def normalize_features(X, train_idxs, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        X_ = scaler.fit_transform(X[train_idxs])
        for idx, i in enumerate(train_idxs):
            X[i] = X_[idx]
        test_idxs = [r for r in range(len(X)) if r not in train_idxs]
        if test_idxs:
            X_ = scaler.transform(X[test_idxs])
            for idx, i in enumerate(test_idxs):
                X[i] = X_[idx]
    else:
        X = scaler.transform(X)
    return X, scaler


def get_pois_by_street(poi_gdf, street_gdf):
    street_index = street_gdf.sindex
    pois_by_street = dict((s, []) for s in range(len(street_gdf)))
    for poi in poi_gdf.itertuples():
        poi_coords = (poi.lon, poi.lat)
        result_street_idx = list(street_index.nearest(poi_coords))[0]
        pois_by_street[result_street_idx].append(poi.Index)
    return pois_by_street


def create_args_dict(poi_gdf, train_idxs, required_args, read_path, write_path):
    args = {'poi_gdf': poi_gdf, 'nlabels': poi_gdf['label'].nunique()}
    if 'label_map' in required_args:
        args['label_map'] = poi_gdf.iloc[train_idxs]['label'].values.tolist()
    if 'geometry_map' in required_args:
        args['geometry_map'] = poi_gdf.iloc[train_idxs]['geometry'].values.tolist()
    if 'poi_index_path' in required_args:
        args['poi_index_path'] = write_path + '/poi_index.pkl'
        af.create_poi_index(poi_gdf.iloc[train_idxs].reset_index(), args['poi_index_path'])
    if 'street_gdf' in required_args:
        street_csv_path = read_path + '/osm_streets.csv'
        args['street_gdf'] = load_street_gdf(street_csv_path)
        args['pois_by_street'] = get_pois_by_street(poi_gdf.iloc[train_idxs].reset_index(), args['street_gdf'])
    if 'textual_index_path' in required_args:
        args['textual_index_path'] = write_path + '/textual_index'
        tf.create_textual_index(poi_gdf.iloc[train_idxs].reset_index(), args['textual_index_path'])
    if 'names' in required_args:
        args['names'] = ' '.join(list(poi_gdf.iloc[train_idxs][config.name_col])).split()
    return args


def create_single_feature(f, args, train_idxs, norm, scaler):
    X = getattr(feature_module_map[f], features_getter_map[f])(
        *[args[arg] for arg in features_getter_args_map[f]])
    if scaler is not None:
        return normalize_features(X, None, scaler)
    elif norm is True:
        return normalize_features(X, train_idxs)
    else:
        return X, None


def create_single_features(poi_gdf, train_idxs, fold_path):
    os.makedirs(fold_path + '/tmp')

    included_features = config.included_adjacency_features + config.included_textual_features
    required_args = set([arg for f in included_features for arg in features_getter_args_map[f]])
    args = create_args_dict(poi_gdf, train_idxs, required_args, os.path.dirname(fold_path), fold_path)

    for f in included_features:
        norm = True if f in config.normalized_features else False
        if f not in features_params_map:
            X, _ = create_single_feature(f, args, train_idxs, norm, None)
            np.save(fold_path + f'/tmp/{f}.npy', X)
        else:
            for p in getattr(config, features_params_map[f]):
                args['param'] = p
                X, _ = create_single_feature(f, args, train_idxs, norm, None)
                np.save(fold_path + f'/tmp/{f}_{p}.npy', X)
    return


def create_concatenated_features(poi_gdf, train_idxs, test_idxs, fold_path):
    included_features = config.included_adjacency_features + config.included_textual_features
    params_names = list(set([features_params_map[f] for f in included_features if f in features_params_map]))
    params_vals = [getattr(config, param) for param in params_names]
    y = poi_gdf['label']

    for idx, params in enumerate(itertools.product(*params_vals)):
        features_params = dict(zip(params_names, params))
        Xs = []
        for f in included_features:
            if f in features_params_map:
                p = features_params[features_params_map[f]]
                Xs.append(np.load(fold_path + f'/tmp/{f}_{p}.npy'))
            else:
                Xs.append(np.load(fold_path + f'/tmp/{f}.npy'))
        X = np.hstack(Xs)
        # X = SelectPercentile(chi2, percentile=75).fit_transform(X, y)
        X_train, X_test = X[train_idxs], X[test_idxs]
        np.save(fold_path + f'/X_train_{idx}.npy', X_train)
        np.save(fold_path + f'/X_test_{idx}.npy', X_test)

    y_train, y_test = y[train_idxs], y[test_idxs]
    np.save(fold_path + '/y_train.npy', y_train)
    np.save(fold_path + '/y_test.npy', y_test)

    path = os.path.dirname(fold_path)
    wrtrs.write_feature_params_info(path + '/params_per_feature_set.csv', params_names, params_vals)
    return


def create_finetuned_features(poi_gdf, features_info, best_feature_params, features_path, results_path):
    included_features = [f[0] for f in features_info]
    required_args = set([arg for f in included_features for arg in features_getter_args_map[f]])
    args = create_args_dict(poi_gdf, np.arange(len(poi_gdf)), required_args, features_path, results_path + '/pickled_objects')

    Xs = []
    for f in features_info:
        feat, norm = f[0], f[1]
        if feat in features_params_map:
            args['param'] = best_feature_params[features_params_map[feat]]
        X, scaler = create_single_feature(feat, args, np.arange(len(poi_gdf)), norm, None)
        if norm is True:
            pickle.dump(scaler, open(results_path + '/pickled_objects' + f'/{feat}_scaler.pkl', 'wb'))
        Xs.append(X)
    X = np.hstack(Xs)
    np.save(results_path + '/X_train.npy', X)
    return X


def create_test_args_dict(test_poi_gdf, required_args, read_path1, read_path2):
    train_poi_gdf = load_poi_gdf(read_path1 + '/train_poi_gdf.csv')
    encoder = pickle.load(open(read_path1 + '/encoder.pkl', 'rb'))
    train_poi_gdf, _ = encode_labels(train_poi_gdf, encoder)

    args = {'poi_gdf': test_poi_gdf, 'nlabels': train_poi_gdf['label'].nunique()}
    if 'label_map' in required_args:
        args['label_map'] = train_poi_gdf['label'].values.tolist()
    if 'geometry_map' in required_args:
        args['geometry_map'] = train_poi_gdf['geometry'].values.tolist()
    if 'poi_index_path' in required_args:
        args['poi_index_path'] = read_path2 + '/poi_index.pkl'
    if 'street_gdf' in required_args:
        street_csv_path = read_path1 + '/osm_streets.csv'
        args['street_gdf'] = load_street_gdf(street_csv_path)
        args['pois_by_street'] = get_pois_by_street(train_poi_gdf, args['street_gdf'])
    if 'textual_index_path' in required_args:
        args['textual_index_path'] = read_path2 + '/textual_index'
    if 'names' in required_args:
        args['names'] = ' '.join(list(train_poi_gdf[config.name_col])).split()
    return args


def create_test_features(poi_gdf, features, features_path, model_training_path, results_path):
    included_features = [f[0] for f in features]
    required_args = set([arg for f in included_features for arg in features_getter_args_map[f]])
    args = create_test_args_dict(poi_gdf, required_args, features_path, model_training_path + '/pickled_objects')

    Xs = []
    for f in features:
        feat, _, param_value, norm = f[0], f[1], f[2], f[3]
        if feat in features_params_map:
            args['param'] = int(param_value) if feature_module_map[feat] == af else float(param_value)
        if norm is True:
            scaler = pickle.load(open(model_training_path + '/pickled_objects' + f'/{feat}_scaler.pkl', 'rb'))
            X, _ = create_single_feature(feat, args, None, norm, scaler)
        else:
            X, _ = create_single_feature(feat, args, None, norm, None)
        Xs.append(X)
    X = np.hstack(Xs)
    np.save(results_path + '/X_test.npy', X)
    return X
