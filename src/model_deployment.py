import pandas as pd
import argparse
import os
import shutil
import pickle
import time

import features_utilities as feat_ut
import clf_utilities as clf_ut
import writers as wrtrs
from ast import literal_eval


def main():
    """
    Implements the fifth step of the experiment pipeline. This step loads a \
    pickled trained model from the previous step and deploys it in order to \
    make predictions on a test dataset.

    Returns:
        None
    """
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-poi_fpath', required=True)
    ap.add_argument('-experiment_path', required=True)
    args = vars(ap.parse_args())

    features_path = args['experiment_path'] + 'features_extraction_results'
    model_training_path = args['experiment_path'] + 'model_training_results'

    for path in [features_path, model_training_path]:
        if os.path.exists(path) is False:
            print('No such file:', path)
            return

    results_path = args['experiment_path'] + 'model_deployment_results'
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)

    t1 = time.time()
    poi_gdf = feat_ut.load_poi_gdf(args['poi_fpath'])
    encoder = pickle.load(open(features_path + '/encoder.pkl', 'rb'))
    poi_gdf, _ = feat_ut.encode_labels(poi_gdf, encoder)

    path = model_training_path + '/features_config.csv'
    features = pd.read_csv(path).values.tolist()

    model_selection_path = args['experiment_path'] + 'model_selection_results'
    path = model_selection_path + '/results_by_clf_params.csv'
    features_selected = literal_eval(pd.read_csv(path, nrows=1).iloc[0, 10])

    X_test = feat_ut.create_test_features(poi_gdf, features, features_path, model_training_path, results_path)
    if features_selected:
        X_test = X_test[:, features_selected]

    model = pickle.load(open(model_training_path + '/model.pkl', 'rb'))

    k_preds = clf_ut.get_top_k_predictions(model, X_test)

    encoder = pickle.load(open(features_path + '/encoder.pkl', 'rb'))
    k_preds = clf_ut.inverse_transform_labels(encoder, k_preds)
    wrtrs.write_predictions(results_path + '/predictions.csv', poi_gdf, k_preds)

    print(f'Model deployment done in {time.time() - t1:.3f} sec.')
    return


if __name__ == "__main__":
    main()
