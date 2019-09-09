import numpy as np
import os
from whoosh.fields import Schema, TEXT, STORED
from whoosh import index, qparser, scoring
from whoosh.analysis import StemmingAnalyzer

import features_utilities as feat_ut
from config import config


def create_textual_index(poi_gdf, path):
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
    top_k_terms = feat_ut.get_top_k(names, k, mode='term')
    X = np.zeros((len(poi_gdf), len(top_k_terms)))
    for poi in poi_gdf.itertuples():
        for t_idx, t in enumerate(top_k_terms):
            if t in getattr(poi, config.name_col):
                X[poi.Index][t_idx] = 1
    return X


def get_top_k_trigrams(poi_gdf, names, k):
    top_k_trigrams = feat_ut.get_top_k(names, k, mode='trigram')
    X = np.zeros((len(poi_gdf), len(top_k_trigrams)))
    for poi in poi_gdf.itertuples():
        for t_idx, t in enumerate(top_k_trigrams):
            if t in getattr(poi, config.name_col):
                X[poi.Index][t_idx] = 1
    return X


def get_top_k_fourgrams(poi_gdf, names, k):
    top_k_fourgrams = feat_ut.get_top_k(names, k, mode='fourgram')
    X = np.zeros((len(poi_gdf), len(top_k_fourgrams)))
    for poi in poi_gdf.itertuples():
        for f_idx, f in enumerate(top_k_fourgrams):
            if f in getattr(poi, config.name_col):
                X[poi.Index][f_idx] = 1
    return X
