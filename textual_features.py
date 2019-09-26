import numpy as np
import os
from whoosh.fields import Schema, TEXT, STORED
from whoosh import index, qparser, scoring
from whoosh.analysis import StemmingAnalyzer

import features_utilities as feat_ut
from config import config


def create_textual_index(poi_gdf, path):
    """
    Creates index containing the pois names given.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois to be stored in the \
            index
        path (str): Path to save the index

    Returns:
        None
    """
    schema = Schema(idx=STORED,
                    name=TEXT(analyzer=StemmingAnalyzer()),
                    label=STORED)
    os.mkdir(path)
    ix = index.create_in(path, schema)
    writer = ix.writer()
    for poi in poi_gdf.itertuples():
        writer.add_document(idx=poi.Index,
                            name=getattr(poi, config.name_col),
                            label=poi.label)
    writer.commit()
    return


def get_similarity_per_class(poi_gdf, textual_index_path, nlabels):
    """
    Creates a features array. For each poi *p* (each row) the array will \
    contain a score in column *c*, representing how similar *p*'s name is \
    with each poi category.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        textual_index_path (str): Path to the stored index
        nlabels (int): Number of poi categories

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), nlabels)
    """
    ix = index.open_dir(textual_index_path)
    X = np.zeros((len(poi_gdf), nlabels))
    with ix.searcher(weighting=scoring.TF_IDF()) as searcher:
        for poi in poi_gdf.itertuples():
            query = qparser.QueryParser('name', ix.schema, group=qparser.OrGroup).parse(getattr(poi, config.name_col))
            results = searcher.search(query)
            for r in results:
                if X[poi.Index][r['label']] < r.score:
                    X[poi.Index][r['label']] = r.score
    return X


def get_top_k_terms(poi_gdf, names, k):
    """
    Creates a features array. Firstly, the top *k* % terms among *names* are \
    considered (e.g. a set of terms *T*). Then, for each poi *p* (each row) \
    the array will contain 1 (True) in column *c*, if term *T[c]* appears in \
    *p*'s name.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        names (list): Contains the names of train pois
        k (float): Percentage of top terms to be considered

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), len(*T*))
    """
    top_k_terms = feat_ut.get_top_k(names, k, mode='term')
    X = np.zeros((len(poi_gdf), len(top_k_terms)))
    for poi in poi_gdf.itertuples():
        for t_idx, t in enumerate(top_k_terms):
            if t in getattr(poi, config.name_col):
                X[poi.Index][t_idx] = 1
    return X


def get_top_k_trigrams(poi_gdf, names, k):
    """
    Creates a features array. Firstly, the top *k* % trigrams among *names* \
    are considered (e.g. a set of trigrams *T*). Then, for each poi *p* (each \
    row) the array will contain 1 (True) in column *c*, if trigram *T[c]* \
    appears in *p*'s name.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        names (list): Contains the names of train pois
        k (float): Percentage of top trigrams to be considered

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), len(*T*))
    """
    top_k_trigrams = feat_ut.get_top_k(names, k, mode='trigram')
    X = np.zeros((len(poi_gdf), len(top_k_trigrams)))
    for poi in poi_gdf.itertuples():
        for t_idx, t in enumerate(top_k_trigrams):
            if t in getattr(poi, config.name_col):
                X[poi.Index][t_idx] = 1
    return X


def get_top_k_fourgrams(poi_gdf, names, k):
    """
    Creates a features array. Firstly, the top *k* % fourgrams among *names* \
    are considered (e.g. a set of fourgrams *T*). Then, for each poi *p* \
    (each row) the array will contain 1 (True) in column *c*, if fourgrams \
    *T[c]* appears in *p*'s name.

    Args:
        poi_gdf (geopandas.GeoDataFrame): Contains pois for which the \
            features will be created
        names (list): Contains the names of train pois
        k (float): Percentage of top fourgrams to be considered

    Returns:
        numpy.ndarray: The features array of shape (n_samples, n_features), \
            here (len(poi_gdf), len(*T*))
    """
    top_k_fourgrams = feat_ut.get_top_k(names, k, mode='fourgram')
    X = np.zeros((len(poi_gdf), len(top_k_fourgrams)))
    for poi in poi_gdf.itertuples():
        for f_idx, f in enumerate(top_k_fourgrams):
            if f in getattr(poi, config.name_col):
                X[poi.Index][f_idx] = 1
    return X
