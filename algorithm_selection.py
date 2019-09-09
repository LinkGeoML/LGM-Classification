import numpy as np
import argparse
import os
import shutil
import time

import clf_utilities as clf_ut
import writers as wrtrs
from config import config


def main():
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-experiment_path', required=True)
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
                X_train = np.load(fold_path + f'/{feature_set[0]}')
                X_test = np.load(fold_path + f'/{feature_set[1]}')
                if clf_name == 'Baseline':
                    by_freq = np.argsort(-np.bincount(y_train))
                    y_pred = np.tile(by_freq, (len(X_test), 1))
                else:
                    clf = clf_ut.train_classifier(clf_name, X_train, y_train)
                    pred = clf.predict_proba(X_test)
                    y_pred = np.argsort(-pred, axis=1)[:, :]
                info = {'fold': i, 'feature_set': feature_set[1], 'classifier': clf_name}
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
