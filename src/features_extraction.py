import os
import datetime
import pickle
import time

from sklearn.model_selection import StratifiedKFold

import features_utilities as feat_ut
import writers as wrtrs
from config import config


def main():
    """
    Implements the first step of the experiment pipeline. Creates a series of \
    X_train, X_test pairs (based on different features parameters \
    combinations) for each one of the folds.

    Returns:
        None
    """
    # Create folder to store experiment
    date_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    exp_path = config.experiments_path + '/exp_' + date_time
    os.makedirs(exp_path)

    # Create folder to store feature extraction results
    results_path = exp_path + '/features_extraction_results'
    os.makedirs(results_path)
    wrtrs.write_feature_space(results_path + '/feature_space.csv')

    # Load pois
    poi_gdf = feat_ut.load_poi_gdf(config.poi_fpath)
    # Shuffle
    poi_gdf = poi_gdf.sample(frac=1).reset_index(drop=True)
    # Remove barely populated labels
    # poi_gdf = poi_gdf.groupby(config.label_col).filter(lambda x: len(x) >= config.n_folds).reset_index(drop=True)

    poi_gdf, encoder = feat_ut.encode_labels(poi_gdf)
    poi_gdf.to_csv(results_path + '/train_poi_gdf.csv', index=False)
    pickle.dump(encoder, open(results_path + '/encoder.pkl', 'wb'))
    poi_ids = list(poi_gdf[config.id_col])
    poi_labels = list(poi_gdf['label'])

    feat_ut.get_required_external_files(poi_gdf, results_path)

    fold = 1
    skf = StratifiedKFold(n_splits=config.n_folds)

    t1 = time.time()
    for train_idxs, test_idxs in skf.split(poi_ids, poi_labels):
        print('Fold:', fold)
        fold_path = results_path + '/fold_' + str(fold)
        os.makedirs(fold_path)

        feat_ut.create_single_features(poi_gdf, train_idxs, fold_path)
        feat_ut.create_concatenated_features(poi_gdf, train_idxs, test_idxs, fold_path)

        fold += 1
    print(f'Feature extraction done in {time.time() - t1:.3f} sec.')
    return


if __name__ == "__main__":
    main()
