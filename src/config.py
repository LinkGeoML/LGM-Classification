class config:

    """
    Class that defines the experiment configuration.

    Attributes:
        poi_fpath (str): Path to csv file containing train pois
        experiments_path (str): Path to folder that stores the experiments

        supported_adjacency_features (list): List of the supported adjacency \
            features to choose from
        supported_textual_features (list): List of the supported textual \
            features to choose from
        included_adjacency_features (list): List of the adjacency features to \
            be included in the experiment
        included_textual_features (list): List of the textual features to \
            be included in the experiment
        normalized_features (list): List of features to be normalized

        classes_in_radius_thr (list): Parameter space for \
            'classes_in_radius_bln' and 'classes_in_radius_cnt' features
        classes_in_street_and_radius_thr (list): Parameter space for \
            'classes_in_street_and_radius_bln' and \
            'classes_in_street_and_radius_cnt' features
        classes_in_neighbors_thr (list): Parameter space for \
            'classes_in_neighbors_bln' and 'classes_in_neighbors_cnt' features

        top_k_terms_pct (list): Parameter space for 'top_k_terms' feature
        top_k_trigrams_pct (list): Parameter space for 'top_k_trigrams' feature
        top_k_fourgrams_pct (list): Parameter space for 'top_k_fourgrams' \
            feature

        n_folds (int): The number of folds in the experiment

        supported_classifiers (list): List of the supported classifiers to \
            choose from
        included_classifiers (list): List of the classifiers to be included \
            in the experiment

        NaiveBayes_hyperparameters (dict): Parameters search space for Naive \
            Bayes classifier
        kNN_hyperparameters (dict): Parameters search space for Nearest \
            Neighbors classifier
        LogisticRegression_hyperparameters (dict): Parameters search space \
            for Logistic Regression classifier
        SVM_hyperparameters (list): Parameters search space for SVM classifier
        MLP_hyperparameters (dict): Parameters search space for MLP classifier
        DecisionTree_hyperparameters (dict): Parameters search space for \
            Decision Tree classifier
        RandomForest_hyperparameters (dict): Parameters search space for \
            Random Forest classifier

        top_k (list): List of different *k*, in order to measure \
            top-*k*-accuracy
        k_preds (int): Number of top predictions to take into consideration
        osm_crs (int): The EPSG crs code that OSM uses

        id_col (str): Column name referring to poi's id
        name_col (str): Column name referring to poi's name
        label_col (str): Column name referring to poi's label
        lon_col (str): Column name referring to poi's longitude
        lat_col (str): Column name referring to poi's latitude
        poi_crs (int): The EPSG crs code used in the pois csv file
    """

    poi_fpath = 'path/to/csv/containing/train/pois'

    experiments_path = 'path/to/folder/holding/experiments'

    supported_adjacency_features = [
        'classes_in_radius_bln', 'classes_in_radius_cnt',
        'classes_in_street_and_radius_bln', 'classes_in_street_and_radius_cnt',
        'classes_in_neighbors_bln', 'classes_in_neighbors_cnt',
        # 'classes_in_street_radius_bln', 'classes_in_street_radius_cnt',
    ]

    supported_textual_features = [
        'similarity_per_class', 'top_k_terms',
        'top_k_trigrams', 'top_k_fourgrams',
    ]

    # supported_geometric_features = [
    #     'area', 'perimeter', 'n_vertices',
    #     'mean_edge_length', 'var_edge_length'
    # ]

    included_adjacency_features = [
        # 'classes_in_radius_bln',
        'classes_in_radius_cnt',
        # 'classes_in_street_and_radius_bln',
        'classes_in_street_and_radius_cnt',
        # 'classes_in_neighbors_bln',
        'classes_in_neighbors_cnt',
        # 'classes_in_street_radius_bln',
        # 'classes_in_street_radius_cnt',
    ]

    included_textual_features = [
        'similarity_per_class',
        'top_k_terms',
        'top_k_trigrams',
        'top_k_fourgrams'
    ]

    # included_geometric_features = [
    #     'area',
    #     'perimeter',
    #     'n_vertices',
    #     'mean_edge_length',
    #     'var_edge_length'
    # ]

    normalized_features = [
        # 'classes_in_radius_cnt',
        # 'classes_in_street_and_radius_cnt',
        # 'classes_in_neighbors_cnt',
        # 'classes_in_street_radius_cnt',
        # 'similarity_per_class'
    ]

    classes_in_radius_thr = [200, 500]
    classes_in_street_and_radius_thr = [300, 500]
    classes_in_neighbors_thr = [5, 20]
    classes_in_street_radius_thr = [100]

    top_k_terms_pct = [0.05]
    top_k_trigrams_pct = [0.01]
    top_k_fourgrams_pct = [0.01]

    # matching_strategy = [
    #     {1: ['within', ['named', 20000], ['avg_lgm_sim_dl', 0.5, None]],
    #      2: ['within', ['unnamed', 10000], [None, None, None]],
    #      3: ['nearby', ['named', 20000], ['lgm_sim_jw', 0.5, 50]],
    #      4: ['nearby', ['unnamed', 10000], [None, None, 50]]}
    # ]

    n_folds = 5

    supported_classifiers = [
        'Baseline',
        'Naive Bayes',
        'Gaussian Process',
        'AdaBoost',
        'Nearest Neighbors',
        'Logistic Regression',
        'SVM',
        'MLP',
        'Decision Tree',
        'Random Forest',
        'Extra Trees'
    ]

    included_classifiers = [
        'Baseline',
        'Naive Bayes',
        # 'Gaussian Process',
        # 'AdaBoost',
        'Nearest Neighbors',
        'Logistic Regression',
        'SVM',
        'MLP',
        'Decision Tree',
        'Random Forest',
        'Extra Trees'
    ]

    NaiveBayes_hyperparameters = {}

    GaussianProcess_hyperparameters = {}

    AdaBoost_hyperparameters = {}

    kNN_hyperparameters = {'n_neighbors': [3, 5, 10]}

    LogisticRegression_hyperparameters = {
        'max_iter': [100, 500],
        'C': [0.1, 1, 10]}

    SVM_hyperparameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]},
        {'kernel': ['poly'], 'degree': [1, 2, 3], 'C': [0.01, 0.1, 1, 10, 100]},
    ]

    MLP_hyperparameters = {
        'hidden_layer_sizes': [(100, ), (50, 50, )],
        'learning_rate_init': [0.0001, 0.01, 0.1],
        'max_iter': [100, 200, 500]}

    DecisionTree_hyperparameters = {
        'max_depth': [1, 4, 16],
        'min_samples_split': [0.1, 0.5, 1.0]}

    RandomForest_hyperparameters = {
        'max_depth': [10, 100, None],
        'n_estimators': [250, 1000]}

    top_k = [1, 5, 10]
    k_preds = 5
    osm_crs = 4326

    # # Marousi
    # id_col = 'poi_id'
    # name_col = 'name'
    # label_col = 'class_name'
    # lon_col = 'x'
    # lat_col = 'y'
    # poi_crs = 2100

    # Yelp
    id_col = 'business_id'
    name_col = 'name'
    label_col = 'category'
    lon_col = 'longitude'
    lat_col = 'latitude'
    poi_crs = 4326
