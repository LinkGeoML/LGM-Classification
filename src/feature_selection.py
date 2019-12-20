import sklearn.feature_selection as sfs
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, RFECV, SelectFromModel
import warnings

with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)

feature_grid_getter_map = {
    'SelectKBest': [chi2, 'selection__k'],
    'VarianceThreshold': ['selection__threshold'],
    'RFE': 'selection__n_features_to_select',
    'SelectFromModel': 'selection__threshold',
    'PCA': 'selection__n_components'
}


def get_stats_features(fsm_name, fsm, clf, params, X_train, y_train, X_test):

    if fsm_name == 'SelectKBest':
        parameters = {}
        parameters['selection__k'] = [int(X_train.shape[1] * per) for per in
                                                 params['selection__k']]
        params = parameters

    pipe = Pipeline([
        ('selection', fsm),
        ('classify', clf)
    ])
    fs_selector = GridSearchCV(estimator=pipe, param_grid=params, cv=4, n_jobs=-1)

    fs_selector.fit(X_train, y_train)
    arguments = [fs_selector.best_params_[arg] if arg in fs_selector.best_params_ else
                 arg for arg in feature_grid_getter_map[fsm_name]]

    fsm_selector = instantianate_feature_selection(fsm_name, arguments)
    X_train = fsm_selector.fit(X_train, y_train).transform(X_train)
    X_test = fsm_selector.transform(X_test)

    return X_train, y_train, X_test, fsm_selector.get_support(indices=True)


def instantianate_feature_selection(fs_name, arguments):
    fs = getattr(sfs, fs_name)
    return fs(*arguments)


def get_RFE_features(clf, clf_name, X_train, y_train, X_test):

    try:
        fs_selector = RFECV(estimator=clf, cv=2, step=0.1, min_features_to_select=85)
        fs_selector.fit(X_train, y_train)
        X_train = fs_selector.transform(X_train)
        X_test = fs_selector.transform(X_test)

        return X_train, y_train, X_test, fs_selector.get_support(indices=True)
    except:
        return X_train, y_train, X_test, []


def get_SFM_features(clf, clf_name, params,  X_train, y_train, X_test):

    try:
        pipe = Pipeline([
            ('selection', SelectFromModel(clf)),
            ('classify', clone(clf))
        ])
        fs_selector = GridSearchCV(pipe, param_grid=params, cv=4, n_jobs=-1)
        fs_selector.fit(X_train, y_train)

        fs_selector = SelectFromModel(estimator=clf, threshold=fs_selector.best_params_['selection__threshold'])
        fs_selector.fit(X_train, y_train)
        X_train = fs_selector.transform(X_train)
        X_test = fs_selector.transform(X_test)
        feature_indices = fs_selector.get_support(indices=True)
        return X_train, y_train, X_test, feature_indices
    except:
        print(f'Select From Model does not support {clf_name}')
        return X_train, y_train, X_test, []


"""def get_PCA_features(clf, params, X_train, y_train, X_test):

    parameters = {}
    parameters['selection__n_components'] = [int(X_train.shape[1]*per) for per in params[0]['selection__n_components']]
    pipe = Pipeline([
        ('selection', PCA()),
        ('classify', clf)
    ])

    fs_selector = GridSearchCV(estimator=pipe, param_grid=parameters, cv=4, n_jobs=-1)
    fs_selector.fit(X_train, y_train)
    fsm_selector = PCA(iterated_power=7, n_components= fs_selector.best_params_['selection__n_components'])
    X_train = fsm_selector.fit_transform(X_train)
    X_test = fsm_selector.transform(X_test)

    return X_train, y_train, X_test, []"""