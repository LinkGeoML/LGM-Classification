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
    Implements the third step of the experiment pipeline. Given a classifier, \
    this step is responsible to find the best performing feature and \
    classifier configuration.

    Returns:
        None
    """
    # Construct argument parser and parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-classifier', required=True)
    ap.add_argument('-experiment_path', required=True)
    args = vars(ap.parse_args())

    if not clf_ut.is_valid(args['classifier']):
        return

    params_grids = clf_ut.clf_hyperparams_map[args['classifier']]
    if isinstance(params_grids, list) is False:
        params_grids = [params_grids]

    t1 = time.time()
    results = []
    for i in range(1, config.n_folds + 1):
        fold_path = args['experiment_path'] + f'features_extraction_results/fold_{i}'
        y_train = np.load(fold_path + '/y_train.npy')
        y_test = np.load(fold_path + '/y_test.npy')
        for params_grid in params_grids:
            for params in clf_ut.create_clf_params_product_generator(params_grid):
                for feature_set in clf_ut.create_feature_sets_generator(fold_path):
                    X_train = np.load(fold_path + f'/{feature_set[0]}')
                    X_test = np.load(fold_path + f'/{feature_set[1]}')
                    clf = clf_ut.clf_callable_map[args['classifier']].set_params(**params)
                    clf.fit(X_train, y_train)
                    # y_pred = clf.predict(X_test)
                    pred = clf.predict_proba(X_test)
                    y_pred = np.argsort(-pred, axis=1)[:, :]
                    info = {'fold': i, 'feature_set': feature_set[1], 'clf_params': str(params)}
                    scores = clf_ut.evaluate(y_test, y_pred)
                    results.append(dict(info, **scores))

    print(f'Model selection done in {time.time() - t1:.3f} sec.')
    results_path = args['experiment_path'] + 'model_selection_results'
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)
    wrtrs.write_classifier_space(results_path + '/classifier_space.csv', args['classifier'])
    wrtrs.write_finetuning_results(results_path, results)
    return


if __name__ == "__main__":
    main()
