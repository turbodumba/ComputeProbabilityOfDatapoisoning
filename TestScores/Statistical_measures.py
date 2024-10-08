import os

import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.svm import OneClassSVM

continuous_methods = ['Z-Score', 'IQR', 'Forest', 'LOF', 'K-Means', 'Mahalanobis']
categorical_methods = ['Frequency_based_outliers', 'Kmodes_outliers', 'DBSCAN_outliers', 'Random_Forest_outliers']


# Helper method converting the given columns into categorical type for the dataframe to work correctly.
def convertToCategorical(df, categorical_columns):
    if len(categorical_columns) > 0:
        for column in categorical_columns:
            if column in df.columns and not isinstance(df[column], pd.Categorical):
                df[column] = df[column].astype('category')


# Helper method to calculate the number of clusters needed for the KModes method.
def determine_n_clusters(num_categories):
    if num_categories == 2:
        return 2  # For binary features
    elif num_categories <= 10:
        return 4  # For low-cardinality categorical features
    else:
        return 8  # For high-cardinality categorical features


# Helper method to calculate the z-score given the data it should be calculated for, in this case a numerical column, and identifying outliers.
def get_z_score_outliers(data, threshold=3):
    # Calculate Z-Scores
    z_scores = np.abs((data - data.mean()) / data.std())
    # Identify outliers
    outliers = z_scores > threshold
    return outliers


# Helper method to calculate the Inter Quartile Range (IQR) for a given column and identifying outliers.
def get_iqr_outliers(data):
    # Calculate IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Identify outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers


# Helper method to run the Isolation Forest algorithm on the given data and identifying outliers.
def get_isolation_forest_outliers(data):
    # Run Isolation Forest algorithm
    clf = IsolationForest(contamination=0.1)
    clf.fit(data.values.reshape(-1, 1))
    y_pred = clf.predict(data.values.reshape(-1, 1))
    # Identify outliers
    outliers = y_pred == -1
    return outliers


# Helper method to run the Local Outlier Factor algorithm on the given data and identifying outliers.
def get_lof_outliers(data):
    # Run Local Outlier Factor algorithm
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred = clf.fit_predict(data.values.reshape(-1, 1))
    # Identify outliers
    outliers = y_pred == -1
    return outliers


# Helper method to find outliers based on the frequency of categorical values.
def get_frequency_based_outliers(data, threshold=0.01):
    # Calculate the frequencies of labels.
    freq = data.value_counts(normalize=True)
    # Identify outliers
    outliers = data.isin(freq[freq < threshold].index)
    return outliers


# Helper method to find outliers using the KModes algorithm.
def get_kmodes_outliers(data, n_clusters=3):
    # Run KModes Algorithm
    data_codes = data.cat.codes.values.reshape(-1, 1)
    km = KModes(n_clusters=n_clusters, init='Huang', n_init=5)
    clusters = km.fit_predict(data_codes)
    cluster_counts = np.bincount(clusters)
    # Identify outliers
    rare_clusters = cluster_counts < 0.05 * len(data_codes)
    if np.any(rare_clusters):
        outlier_cluster = np.where(rare_clusters)[0][0]
        outliers = data[clusters == outlier_cluster]
        outliers = [True for _ in outliers]
    else:
        outliers = np.array([])
    return outliers


# Helper method to find outliers using the KMeans algorithm.
def get_kmeans_outliers(data, n_clusters=8, threshold=0.05):
    # Run KMeans algorithm
    data_reshaped = data.values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data_reshaped)
    cluster_counts = np.bincount(clusters)
    # Identify outliers
    rare_clusters = cluster_counts < threshold * len(data)
    outliers = data[np.isin(clusters, np.where(rare_clusters)[0])]
    outliers = [True for _ in outliers]
    return outliers


# Helper method to find outliers using the Random Forest algorithm.
def get_random_forest_outliers(df, n_estimators=150):
    # Run Random Forest Classifier algorithm
    rf = RandomForestClassifier(n_estimators=n_estimators)
    X = pd.get_dummies(df)
    rf.fit(X, np.ones(X.shape[0]))  # Train on normal data
    leaf_indices = rf.apply(X)  # Get leaf indices for each data point
    leaf_counts = np.zeros((X.shape[0], rf.n_estimators))

    # Identify outliers
    for tree_idx, tree in enumerate(rf.estimators_):
        for leaf_idx in range(X.shape[0]):
            leaf_counts[leaf_idx, tree_idx] = np.sum(leaf_indices[:, tree_idx] == leaf_indices[leaf_idx, tree_idx])
    avg_depth = np.mean(leaf_counts, axis=1)
    outlier_threshold = np.percentile(avg_depth, 90)
    outliers = avg_depth > outlier_threshold
    outliers = [True for _ in outliers]
    return outliers


# Helper method to find outliers using the Density-based spatial clustering of applications noise method.
def get_dbscan_outliers(df, eps=0.5, min_samples=5):
    df = df.to_frame()
    # Convert categorical data to one-hot encoded format
    encoder = OneHotEncoder()
    data_encoded = encoder.fit_transform(df).toarray()

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='hamming')
    clusters = dbscan.fit_predict(data_encoded)

    # Identify outliers
    outliers = df[clusters == -1]
    outliers = [True for _ in outliers]
    return outliers


# Helper method to find outliers using the Mahalanobis outliers method.
def get_mahalanobis_outliers(df, threshold=3):
    # Preprocess data for the calculation
    df = np.array(df)
    if df.ndim == 1:
        df = df.reshape(-1, 1)
    # Calculate the mahalanobis distances
    mean = np.mean(df, axis=0)
    cov = np.cov(df, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[cov]])  # Ensure the covariance matrix is 2D
    inv_cov = np.linalg.inv(cov)
    diff = df - mean
    md = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    # Identify outliers
    outliers = md > threshold
    return outliers


# Helper method used to preprocess data for the ensemble method to work on the whole dataset.
def preprocess_data(df, categorical_columns):
    # Apply one-hot encoding to categorical columns.
    df = pd.get_dummies(df, columns=categorical_columns)

    # Ensure all columns are numeric and convert non numeric columns.
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
        elif df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # Drop rows with missing values
    df = df.dropna()
    return df


# Helper method to detect identify outliers using the Ensemble Method, which is a combination of Isolation Forest, One class SVM and Local outlier factor
def get_ensemble_outliers(df, categorical_columns, contamination=0.1):
    df = preprocess_data(df, categorical_columns)
    # Initialize models
    iso_forest = IsolationForest(contamination=contamination)
    one_class_svm = OneClassSVM(nu=contamination)
    lof = LocalOutlierFactor(n_neighbors=int(np.sqrt(len(df))), contamination=contamination)

    # Fit models
    iso_forest.fit(df)
    one_class_svm.fit(df)
    lof.fit(df)

    # Get anomaly scores
    iso_scores = -iso_forest.decision_function(df)
    svm_scores = -one_class_svm.decision_function(df)
    lof_scores = -lof.negative_outlier_factor_

    # Normalize scores
    scaler = MinMaxScaler()
    iso_scores = scaler.fit_transform(iso_scores.reshape(-1, 1))
    svm_scores = scaler.fit_transform(svm_scores.reshape(-1, 1))
    lof_scores = scaler.fit_transform(lof_scores.reshape(-1, 1))

    # Combine scores
    combined_scores = (iso_scores + svm_scores + lof_scores) / 3

    # Identify outliers
    threshold = 0.25
    anomalies = combined_scores > threshold
    return anomalies.flatten().tolist()


# Helper method which is in charge of coordinating the detection of outliers by all the different metrics shown.
def detect_outliers(df, categoricalColumns, numericalColumns, z_threshold=3, binary_threshold=0.3,
                    category_threshold_multiple=0.5,
                    high_cardinality_threshold=0.01, categorical_unique_value_threshold=10, rF_n_estimators=150,
                    dbscan_eps=0.5, dbscan_min_samples=5, ensemble_contamination=0.1, kMeans_n_clusters=3,
                    kMeans_threshold=0.05):
    outliers = {}

    # Iterating over numerical columns given as a parameter.
    for column in numericalColumns:
        # Dropping NaN values for the metrics to work correctly.
        data = df[column].dropna()

        # Detecting the outliers separately using their respective helper methods.
        z_score_outliers = get_z_score_outliers(data, threshold=z_threshold)
        iqr_outliers = get_iqr_outliers(data)
        isolation_forest_outliers = get_isolation_forest_outliers(data)
        lof_outliers = get_lof_outliers(data)
        kmeans_outliers = get_kmeans_outliers(data, kMeans_n_clusters, kMeans_threshold)

        outliers[column] = {
            'Z-Score': z_score_outliers,
            'IQR': iqr_outliers,
            'Forest': isolation_forest_outliers,
            'LOF': lof_outliers,
            'K-Means': kmeans_outliers,
        }

    # Iterating over categorical columns given as a parameter.
    for column in categoricalColumns:
        value_counts = df[column].value_counts(normalize=True)
        num_categories = len(value_counts)
        # Adjust threshold for the frequency based method based on number of categories.
        if num_categories == 2:
            threshold = binary_threshold  # For binary features
        elif num_categories <= categorical_unique_value_threshold:
            threshold = 1 / num_categories * category_threshold_multiple  # For low-cardinality categorical features
        else:
            threshold = high_cardinality_threshold  # For high-cardinality categorical features

        # Detecting the outliers separately using their respective helper methods.
        frequency_based_outliers = get_frequency_based_outliers(df[column], threshold=threshold)
        kmodes_outliers = get_kmodes_outliers(df[column], n_clusters=determine_n_clusters(num_categories))
        random_forest_outliers = get_random_forest_outliers(df[column], rF_n_estimators)
        dbscan_outliers = get_dbscan_outliers(df[column], dbscan_eps, dbscan_min_samples)

        outliers[column] = {
            'Frequency_based_outliers': frequency_based_outliers,
            'Kmodes_outliers': kmodes_outliers,
            'Random_Forest_outliers': random_forest_outliers,
            'DBSCAN_outliers': dbscan_outliers
        }

    # Identifying the outliers based on the mahalanobis and ensemble helper methods.
    mahalanobis_outliers = get_mahalanobis_outliers(df[numericalColumns], z_threshold)
    ensemble_outliers = get_ensemble_outliers(df, categoricalColumns, ensemble_contamination)
    return outliers, ensemble_outliers, mahalanobis_outliers


# Helper method to calculate the percentage of outliers contained in the dataset, given the two inputs.
def calculate_percentage(anomalies, total):
    percentage = len(anomalies) / total
    if percentage < 0.025:
        return 0
    elif percentage > 0.4:
        return 1
    else:
        return (percentage - 0.03) / (0.4 - 0.03)


# Helper function to categorize the different metrics into the correct category for further use.
def categorize_outliers(measures):
    continuous_outliers = {}
    categorical_outliers = {}

    for column, methods_in_column in measures.items():
        for method, anomalies in methods_in_column.items():
            if method in continuous_methods:  # Continuous
                if column not in continuous_outliers:
                    continuous_outliers[column] = {}
                continuous_outliers[column][method] = anomalies
            elif method in categorical_methods:  # Categorical
                if column not in categorical_outliers:
                    categorical_outliers[column] = {}
                categorical_outliers[column][method] = anomalies

    return continuous_outliers, categorical_outliers


# Helper function to calculate the separate score for each different category provided.
def calculate_category_score(dataset, outliers, total_columns, scores):
    for column, methods_in_column in outliers.items():
        total = len(dataset[column])
        for method, anomalies in methods_in_column.items():
            actual_anomalies = [a for a in anomalies if a]
            score = calculate_percentage(actual_anomalies, total)
            scores.append(score)
    return sum(scores) / total_columns if total_columns > 0 else 0


# Helper method coordinating the whole scoring process of the algorithm, calculating the seperate stores and combining them correctly.
def get_score(dataset, measures, ensemble_outliers, mahalanobis_outliers, categoricalColumns, numericalColumns):
    num_categorical = len(categoricalColumns)
    num_numerical = len(numericalColumns)
    total_columns = num_categorical + num_numerical

    # Calculating the different weights.
    categorical_weight = num_categorical / total_columns if total_columns > 0 else 0
    numerical_weight = num_numerical / total_columns if total_columns > 0 else 0

    # Categorizing the measures with the helper function.
    continuous_outliers, categorical_outliers = categorize_outliers(measures)

    mahalanobis_score = calculate_percentage([a for a in mahalanobis_outliers if a],
                                             sum((dataset[numericalColumns]).apply(len)))

    # Calculate scores for each category
    categorical_scores = calculate_category_score(dataset, categorical_outliers, len(categoricalColumns), [])
    continuous_scores = calculate_category_score(dataset, continuous_outliers, len(numericalColumns),
                                                 [mahalanobis_score])

    actual_ensemble_outliers = [a for a in ensemble_outliers if a]
    ensemble_score = calculate_percentage(actual_ensemble_outliers, len(dataset))

    # Combing the different scores calculated into one final score.
    combined_score = (
                             numerical_weight * continuous_scores + categorical_weight * categorical_scores + ensemble_score) / 3
    # Normalizing the combined score.
    normalized_score = min(max(combined_score, 0), 1)
    return normalized_score


# The main function of the statistical measures class, which starts the whole scoring process.
def calculateStatistics(filepath, categoricalColumns, numericalColumns, z_threshold=3, binary_threshold=0.3,
                        category_threshold_multiple=0.5,
                        high_cardinality_threshold=0.01, categorical_unique_value_threshold=10, rF_n_estimators=150,
                        dbscan_eps=0.5, dbscan_min_samples=5, ensemble_contamination=0.1, kMeans_n_clusters=3,
                        kMeans_threshold=0.05):
    # Finding the absolute path of the file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(base_dir, 'Datasets')
    # Change directory to the dataset folder
    os.chdir(datasets_dir)
    # Read the CSV file
    dataset = pd.read_csv(filepath)
    # Convert the categorical columns to make sure they are saved correctly.
    convertToCategorical(dataset, categoricalColumns)
    # Calculate the outliers of the different measures using the detect outliers helper function.
    measures, ensemble_outliers, mahalanobis_outliers = detect_outliers(dataset, categoricalColumns, numericalColumns,
                                                                        z_threshold, binary_threshold,
                                                                        category_threshold_multiple,
                                                                        high_cardinality_threshold,
                                                                        categorical_unique_value_threshold,
                                                                        rF_n_estimators, dbscan_eps, dbscan_min_samples,
                                                                        ensemble_contamination, kMeans_n_clusters,
                                                                        kMeans_threshold)
    # Calculate the final score of the different metrics using the get score helper function.
    return get_score(dataset, measures, ensemble_outliers, mahalanobis_outliers, categoricalColumns, numericalColumns)
