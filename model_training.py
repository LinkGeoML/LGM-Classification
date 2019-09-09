import pandas as pd
import argparse
from ast import literal_eval
import os
import shutil
import pickle
import time

import features_utilities as feat_ut
import clf_utilities as clf_ut
import writers as wrtrs


def main():
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-experiment_path', required=True)
    args = vars(ap.parse_args())

    features_path = args['experiment_path'] + 'features_extraction_results'
    model_selection_path = args['experiment_path'] + 'model_selection_results'

    for path in [model_selection_path, features_path]:
        if os.path.exists(path) is False:
            print('No such file:', path)
            return

    results_path = args['experiment_path'] + 'model_training_results'
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    os.mkdir(results_path + '/pickled_objects')

    t1 = time.time()
    path = model_selection_path + '/classifier_space.csv'
    clf_name = list(pd.read_csv(path, skiprows=1, nrows=1))[0]

    path = model_selection_path + '/results_by_feature_and_clf_params.csv'
    best_config = list(pd.read_csv(path, skiprows=1, nrows=1))

    nbest_feature_set = int(best_config[0].split('_')[-1].split('.')[0])
    best_clf_params = literal_eval(best_config[1])

    path = features_path + '/params_per_feature_set.csv'
    best_feature_params = literal_eval(list(pd.read_csv(path, skiprows=nbest_feature_set+1, nrows=1))[1])

    poi_gdf = feat_ut.load_poi_gdf(features_path + '/train_poi_gdf.csv')
    encoder = pickle.load(open(features_path + '/encoder.pkl', 'rb'))
    poi_gdf, _ = feat_ut.encode_labels(poi_gdf, encoder)

    path = features_path + '/feature_space.csv'
    features_info = pd.read_csv(path)[['Feature', 'Normalized']].values.tolist()

    X_train = feat_ut.create_finetuned_features(poi_gdf, features_info, best_feature_params, features_path, results_path)
    y_train = list(poi_gdf['label'])

    model = clf_ut.clf_callable_map[clf_name].set_params(**best_clf_params)
    model.fit(X_train, y_train)

    pickle.dump(model, open(results_path + '/model.pkl', 'wb'))

    print(f'Model training done in {time.time() - t1:.3f} sec.')
    wrtrs.write_feature_space(results_path + '/features_config.csv', features_info, best_feature_params)
    wrtrs.write_classifier_space(results_path + '/classifier_config.csv', clf_name, best_clf_params)
    return


if __name__ == "__main__":
    main()
