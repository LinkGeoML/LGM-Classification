class config:

    poi_fpath = '/media/disk/LGM-PC-utils/data/toronto/yelp_toronto_train.csv'

    experiments_path = '/media/disk/LGM-PC-utils/experiments'

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
