import pandas as pd
from itertools import product
import csv

import features_utilities as feat_ut
import clf_utilities as clf_ut
from config import config


def write_feature_params_info(fpath, params_names, params_vals):
    """
    Creates a csv file defining the features parameters used in each of the \
    features sets.

    Args:
        fpath (str): File path to write
        params_names (list): Contains the features parameters names
        params_vals (list): Contains a list with values for each feature \
            parameter

    Returns:
        None
    """
    params_info = {}
    for idx, params in enumerate(product(*params_vals)):
        features_params = dict(zip(params_names, params))
        params_info[f'X_train_{idx}, X_test_{idx}'] = features_params
    with open(fpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Feature sets', 'Feature parameters combination'])
        for k, v in params_info.items():
            writer.writerow([k, v])
    return


def write_feature_space(fpath, features_info=None, best_params=None):
    """
    Creates a csv file containing the features configuration to be searched \
    (namely included features as well as their parameters to be finetuned and \
    whether normalizaton should be used or not). If *best_params* is given, \
    then the csv will present the finetuned features parameters values instead.

    Args:
        fpath (str): File path to write
        features_info (list, optional): Contains (features, normalized or \
            not) pairs
        best_params (dict, optional): Contains features as keys and the \
            corresponding finetuned values as values

    Returns:
        None
    """
    with open(fpath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature', 'Parameter', 'Parameter values', 'Normalized'])
        if features_info is None:
            included_features = config.included_adjacency_features + config.included_textual_features
            for feat in included_features:
                if feat not in feat_ut.features_params_map:
                    writer.writerow([feat, '-', '-', True if feat in config.normalized_features else False])
                else:
                    param_name = feat_ut.features_params_map[feat]
                    writer.writerow([feat, param_name, getattr(config, param_name), True if feat in config.normalized_features else False])
        else:
            for f in features_info:
                feat, norm = f[0], f[1]
                if feat not in feat_ut.features_params_map:
                    writer.writerow([feat, '-', '-', norm])
                else:
                    param_name = feat_ut.features_params_map[feat]
                    writer.writerow([feat, param_name, best_params[param_name], norm])
    return


def write_classifier_space(fpath, clf_name, best_params=None):
    """
    Creates a csv file containing the chosen classifier's name as well as the \
    hyperparameters space to be searched. If *best_params* is given, then the \
    csv will present the finetuned hyperparameters values instead.

    Args:
        fpath (str): File path to write
        clf_name (str): Name of the utilized classifier
        best_params (dict, optional): Contains hyperparameters names as keys \
            and the corresponding finetuned values as values

    Returns:
        None
    """
    with open(fpath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Classifier', 'Parameters'])
        if best_params is None:
            writer.writerow([clf_name, clf_ut.clf_hyperparams_map[clf_name]])
        else:
            writer.writerow([clf_name, best_params])
    return


def write_evaluation_space(fpath):
    """
    Creates a csv file that contains all the included classifiers as well as \
    the hyperparameters space to be searched for each of them.

    Args:
        fpath (str): File path to write

    Returns:
        None
    """
    with open(fpath, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Classifier', 'Parameters'])
        for clf in config.included_classifiers:
            if clf not in clf_ut.clf_hyperparams_map:
                writer.writerow([clf, '-'])
            else:
                writer.writerow([clf, clf_ut.clf_hyperparams_map[clf]])
    return


def write_evaluation_results(results_path, results_dict):
    """
    Creates three csv files that present (a) full results (b) results per \
    fold and (c) results per classifier of the algorithm_selection step.

    Args:
        results_path (str): Path to write
        results_dict (dict): Contains the algorithm_selection step's results

    Returns:
        None
    """
    all_results_df = pd.DataFrame(results_dict)
    all_results_df.to_csv(
        results_path + '/all_results.csv',
        columns=['fold', 'feature_set', 'classifier',
                 'top_1_accuracy', 'top_5_accuracy', 'top_10_accuracy',
                 'f1_macro', 'f1_micro', 'f1_weighted',
                 'precision_weighted', 'recall_weighted'],
        index=False)

    fold_results_df = all_results_df.groupby(['fold', 'classifier']).mean()
    fold_results_df.to_csv(results_path + '/results_by_fold.csv')

    clf_results_df = fold_results_df.groupby(['fold', 'classifier']).sum().groupby(level=1).mean()
    clf_results_df.sort_values(by=['f1_weighted'], ascending=False).to_csv(results_path + '/results_by_classifier.csv')
    return


def write_finetuning_results(results_path, results_dict):
    """
    Creates two csv files that present (a) full results and (b) results per \
    feature set and classifier hyperparameters used of the model_selection \
    step.

    Args:
        results_path (str): Path to write
        results_dict (dict): Contains the model_selection step's results

    Returns:
        None
    """
    all_results_df = pd.DataFrame(results_dict)
    all_results_df.to_csv(
        results_path + '/all_results.csv',
        columns=['fold', 'feature_set', 'clf_params',
                 'top_1_accuracy', 'top_5_accuracy', 'top_10_accuracy',
                 'f1_macro', 'f1_micro', 'f1_weighted',
                 'precision_weighted', 'recall_weighted'],
        index=False)

    avg_results_df = all_results_df.groupby(['feature_set', 'clf_params']).mean()
    avg_results_df = avg_results_df.drop('fold', 1)
    avg_results_df.sort_values(by=['f1_weighted'], ascending=False).to_csv(results_path + '/results_by_feature_and_clf_params.csv')
    return


def write_predictions(fpath, poi_gdf, k_preds):
    """
    Creates a csv file to present the top k predictions (in (predicted label, \
    score) pairs).

    Args:
        fpath (str): File path to write
        poi_gdf (geopandas.GeoDataFrame): Contains the pois to which the \
            predictions refer to
        k_preds (list): Contains (predicted label, score) pairs

    Returns:
        None
    """
    with open(fpath, 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            config.id_col,
            config.name_col,
            f'top_{config.k_preds}_predictions'])
        for poi in poi_gdf.itertuples():
            writer.writerow([
                getattr(poi, config.id_col),
                getattr(poi, config.name_col),
                [
                    k_pred
                    for k_pred in k_preds[poi.Index *
                                          config.k_preds:poi.Index *
                                          config.k_preds + config.k_preds]
                ]
            ])
    return
