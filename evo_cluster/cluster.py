import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def cluster_data(df, columns, n_clusters, wcss_min_max, d_min_max, alpha=0.2, beta=0.2, gamma=0.6):
    """
    Clusters the given DataFrame using the specified columns and number of clusters.
    Calculates a composite score based on silhouette score, WCSS, and inter-cluster distance.

    Args:
    - df: DataFrame containing the data to be clustered.
    - columns: List of column names to use for clustering.
    - n_clusters: Number of clusters to form.
    - wcss_min_max: Tuple containing min and max WCSS.
    - d_min_max: Tuple containing min and max distances.
    - alpha, beta, gamma: Weights for the silhouette score, WCSS, and inter-cluster distance in the composite score.
    Returns:
    - composite_score: Calculated composite score for the clustering.
    """

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(df[columns])
    labels = kmeans.labels_
    
    df['CLUSTER'] = labels
    silhouette_avg = silhouette_score(df[columns], labels)

    wcss = kmeans.inertia_

    centroids = kmeans.cluster_centers_
    centroid_distances = cdist(centroids, centroids, 'euclidean')
    average_centroid_distance = np.mean(centroid_distances[np.triu_indices_from(centroid_distances, k=1)])

    # Normalize the metrics
    s_norm = (silhouette_avg + 1) / 2  # Silhouette score ranges from -1 to 1
    wcss_norm = (wcss - wcss_min_max[0]) / (wcss_min_max[1] - wcss_min_max[0])
    d_norm = (average_centroid_distance - d_min_max[0]) / (d_min_max[1] - d_min_max[0])

    wcss_norm = np.clip(wcss_norm, 0, 1)
    d_norm = np.clip(d_norm, 0, 1)

    # Calculate the composite score
    composite_score = ((s_norm * alpha) - (beta * wcss_norm) + (gamma * d_norm))
    return df, composite_score

def estimate_wcss_and_distance_ranges(df, columns, cluster_range, num_trials=40):
    """
    Estimates the ranges for WCSS and inter-cluster distances based on multiple clustering trials.

    Args:
    - df: DataFrame to operate on.
    - columns: List of all possible columns to include in trials.
    - cluster_range: Tuple containing min and max number of clusters.
    - num_trials: Number of trials to perform.
    Returns:
    - Tuple of (min_max_wcss, min_max_distances)
    """
    wcss_values = []
    distance_values = []
    
    column_indices = [df.columns.get_loc(col) for col in columns if col in df.columns]

    max_num_columns = min(len(columns), max(cluster_range))
    
    for _ in range(num_trials):
        num_columns = random.randint(2, max_num_columns)
        selected_columns_indices = random.sample(column_indices, num_columns)
        selected_columns = [df.columns[i] for i in selected_columns_indices]

        n_clusters = random.randint(cluster_range[0], cluster_range[1])
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(df[selected_columns])

        wcss = kmeans.inertia_
        wcss_values.append(wcss)

        if n_clusters > 1:
            centroids = kmeans.cluster_centers_
            centroid_distances = cdist(centroids, centroids, 'euclidean')
            avg_distance = np.mean(centroid_distances[np.triu_indices_from(centroid_distances, k=1)])
            distance_values.append(avg_distance)

    wcss_min_max = (min(wcss_values), max(wcss_values))
    d_min_max = (min(distance_values), max(distance_values)) if distance_values else (0, 1)
    return wcss_min_max, d_min_max
