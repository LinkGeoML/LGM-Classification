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
    train_sets = [f for f in os.listdir(fold_path) if f.startswith('X_train_')]
    train_sets = sorted(train_sets, key=lambda i: (len(i), i))

    test_sets = [f for f in os.listdir(fold_path) if f.startswith('X_test_')]
    test_sets = sorted(test_sets, key=lambda i: (len(i), i))

    feature_sets = zip(train_sets, test_sets)
    for feature_set in feature_sets:
        yield feature_set


def train_classifier(clf_name, X_train, y_train):
    clf = clf_callable_map[clf_name]
    params = clf_hyperparams_map[clf_name]
    clf = GridSearchCV(clf, params, cv=4, scoring='f1_weighted', n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def k_accuracy_score(y_test, k_best):
    c = 0
    for i, y in enumerate(y_test):
        if y in k_best[i]:
            c += 1
    return c / (i + 1)


def evaluate(y_test, y_pred):
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
    supported_clfs = [
        clf for clf in config.supported_classifiers if clf != 'Baseline'
    ]
    if clf_name not in supported_clfs:
        print('Supported classifiers:', supported_clfs)
        return False
    return True


def create_clf_params_product_generator(params_grid):
    keys = params_grid.keys()
    vals = params_grid.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def normalize_scores(scores):
    s = sum(scores)
    normalized = [score/s for score in scores]
    return normalized


def get_top_k_predictions(model, X_test):
    preds = model.predict_proba(X_test)
    k_preds = []
    for pred in preds:
        k_labels = np.argsort(-pred)[:config.k_preds]
        k_scores = normalize_scores(pred[k_labels])
        k_preds.append(zip(k_labels, k_scores))
    return k_preds


def inverse_transform_labels(encoder, k_preds):
    label_mapping = dict(
        zip(encoder.transform(encoder.classes_), encoder.classes_))
    k_preds_new = [(label_mapping[pred[0]], pred[1]) for k_pred in k_preds
                   for pred in k_pred]
    return k_preds_new
