import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import config


clf_callable_map = {
    'Naive Bayes': GaussianNB(),
    'Gaussian Process': GaussianProcessClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(solver='liblinear', multi_class='auto'),
    'SVM': SVC(probability=True),
    'MLP': MLPClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Extra Trees': ExtraTreesClassifier()}

clf_hyperparams_map = {
    'Naive Bayes': config.NaiveBayes_hyperparameters,
    'Gaussian Process': config.GaussianProcess_hyperparameters,
    'AdaBoost': config.AdaBoost_hyperparameters,
    'Nearest Neighbors': config.kNN_hyperparameters,
    'Logistic Regression': config.LogisticRegression_hyperparameters,
    'SVM': config.SVM_hyperparameters,
    'MLP': config.MLP_hyperparameters,
    'Decision Tree': config.DecisionTree_hyperparameters,
    'Random Forest': config.RandomForest_hyperparameters,
    'Extra Trees': config.RandomForest_hyperparameters}


def create_feature_sets_generator(fold_path):
    """
    Creates a generator yielding features sets names.

    Args:
        fold_path (str): Path to read features sets

    Yields:
        list: pairs of X_train, X_test features sets names
    """
    train_sets = [f for f in os.listdir(fold_path) if f.startswith('X_train_')]
    train_sets = sorted(train_sets, key=lambda i: (len(i), i))

    test_sets = [f for f in os.listdir(fold_path) if f.startswith('X_test_')]
    test_sets = sorted(test_sets, key=lambda i: (len(i), i))

    feature_sets = zip(train_sets, test_sets)
    for feature_set in feature_sets:
        yield feature_set


def train_classifier(clf_name, X_train, y_train):
    """
    Trains a classifier through grid search.

    Args:
        clf_name (str): Classifier's name to be trained
        X_train (numpy.ndarray): Train features array
        y_train (numpy.ndarray): Train labels array

    Returns:
        object: The trained classifier
    """
    clf = clf_callable_map[clf_name]
    params = clf_hyperparams_map[clf_name]
    clf = GridSearchCV(clf, params, cv=4, scoring='f1_weighted', n_jobs=-1, iid=True)
    clf.fit(X_train, y_train)
    return clf


def k_accuracy_score(y_test, k_best):
    """
    Measures the defined k-accuracy metric. For each poi, a successful \
    prediction is considered if true label appears in the top k labels \
    predicted by the model,

    Args:
        y_test (numpy.ndarray): True labels
        k_best (numpy.ndarray): Top k predicted labels

    Returns:
        float: The k accuracy score
    """
    c = 0
    for i, y in enumerate(y_test):
        if y in k_best[i]:
            c += 1
    return c / (i + 1)


def evaluate(y_test, y_pred):
    """
    Evaluates model predictions through a series of metrics.

    Args:
        y_test (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted labels

    Returns:
        dict: Contains metrics names as keys and the corresponding values as \
        values
    """
    scores = {}
    for k in config.top_k:
        k_best = y_pred[:, :k]
        scores[f'top_{k}_accuracy'] = k_accuracy_score(y_test, k_best)

    y_pred = y_pred[:, :1]
    scores['f1_macro'] = f1_score(y_test, y_pred, average='macro')
    scores['f1_micro'] = f1_score(y_test, y_pred, average='micro')
    scores['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
    scores['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
    scores['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
    return scores


def is_valid(clf_name):
    """
    Checks whether *clf_name* is a valid classifier's name with respect to \
    the experiment setup.

    Args:
        clf_name (str): Classifier's name

    Returns:
        bool: Returns True if given classifier's name is valid
    """
    supported_clfs = [
        clf for clf in config.supported_classifiers if clf != 'Baseline'
    ]
    if clf_name not in supported_clfs:
        print('Supported classifiers:', supported_clfs)
        return False
    return True


def create_clf_params_product_generator(params_grid):
    """
    Generates all possible combinations of classifier's hyperparameters values.

    Args:
        params_grid (dict): Contains classifier's hyperparameters names as \
            keys and the correspoding search space as values

    Yields:
        dict: Contains a classifier's hyperparameters configuration
    """
    keys = params_grid.keys()
    vals = params_grid.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def normalize_scores(scores):
    """
    Normalizes predictions scores to a probabilities-like format.

    Args:
        scores (list): Contains the predictions scores as predicted by the \
            model

    Returns:
        list: The normalized scores
    """
    s = sum(scores)
    normalized = [score/s for score in scores]
    return normalized


def get_top_k_predictions(model, X_test):
    """
    Makes predictions utilizing *model* over *X_test*.

    Args:
        model (object): The model to be used for predictions
        X_test (numpy.ndarray): The test features array

    Returns:
        list: Contains predictions in (label, score) pairs
    """
    preds = model.predict_proba(X_test)
    k_preds = []
    for pred in preds:
        k_labels = np.argsort(-pred)[:config.k_preds]
        k_scores = normalize_scores(pred[k_labels])
        k_preds.append(zip(k_labels, k_scores))
    return k_preds


def inverse_transform_labels(encoder, k_preds):
    """
    Utilizes *encoder* to transform encoded labels back to the original \
    strings.

    Args:
        encoder (sklearn.preprocessing.LabelEncoder): The encoder to be \
            utilized
        k_preds (list): Contains predictions in (label, score) pairs

    Returns:
        list: Contains predictions in (label, score) pairs, where label is \
            now in the original string format
    """
    label_mapping = dict(
        zip(encoder.transform(encoder.classes_), encoder.classes_))
    k_preds_new = [(label_mapping[pred[0]], pred[1]) for k_pred in k_preds
                   for pred in k_pred]
    return k_preds_new
