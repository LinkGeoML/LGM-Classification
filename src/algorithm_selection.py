import numpy as np
import argparse
import os
import shutil
import time

import clf_utilities as clf_ut
import writers as wrtrs
from config import config


def main():
    """
    Implements the second step of the experiment pipeline. Trains a series of \
    classifiers based on different configurations in terms of both features \
    and classifiers hyperparameters in a nested cross validation scheme.

    Returns:
        None
    """
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-experiment_path', required=True)
    ap.add_argument('-feature_selection', required=False, action='store_true')
    ap.add_argument('-fs_method', required=False)
    args = vars(ap.parse_args())

    t1 = time.time()
    results = []
    for i in range(1, config.n_folds + 1):
        print('Fold:', i)
        fold_path = args['experiment_path'] + f'features_extraction_results/fold_{i}'
        y_train = np.load(fold_path + '/y_train.npy')
        y_test = np.load(fold_path + '/y_test.npy')
        for clf_name in config.included_classifiers:
            print('Classifier:', clf_name)
            for feature_set in clf_ut.create_feature_sets_generator(fold_path):
                X_train_all = np.load(fold_path + f'/{feature_set[0]}')
                X_test_all = np.load(fold_path + f'/{feature_set[1]}')
                if clf_name == 'Baseline':
                    by_freq = np.argsort(-np.bincount(y_train))
                    y_pred = np.tile(by_freq, (len(X_test_all), 1))
                else:
                    if args['feature_selection']:
                        X_train, y_train, X_test, feature_indices = clf_ut.ft_selection(clf_name, args['fs_method'], X_train_all, y_train, X_test_all)
                    else:
                        X_train, X_test = X_train_all, X_test_all
                        feature_indices = []
                    clf = clf_ut.train_classifier(clf_name, X_train, y_train)
                    pred = clf.predict_proba(X_test)
                    y_pred = np.argsort(-pred, axis=1)[:, :]
                info = {'fold': i, 'feature_set': feature_set[1], 'classifier': clf_name,
                        'feature_col':list(feature_indices)}
                scores = clf_ut.evaluate(y_test, y_pred)
                results.append(dict(info, **scores))

    print(f'Algorithm selection done in {time.time() - t1:.3f} sec.')
    results_path = args['experiment_path'] + 'algorithm_selection_results'
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    wrtrs.write_evaluation_space(results_path + '/evaluation_space.csv')
    wrtrs.write_evaluation_results(results_path, results)
    return


if __name__ == "__main__":
    main()
