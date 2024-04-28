from .validation import validate_input
from .algo import evolutionary_algorithm
from .cluster import estimate_wcss_and_distance_ranges

class EvoCluster:
    """
    EvoCluster performs clustering using an evolutionary algorithm approach to optimize both
    feature selection and the number of clusters. The clustering evaluation is based on silhouette score, 
    within-cluster sum of squares (WCSS), and the distance between cluster centroids.

    Attributes:
        df (pd.DataFrame): The DataFrame on which clustering is performed.
        columns (list of str): List of column names used for clustering.
        min_clusters (int): Minimum number of clusters to consider.
        max_clusters (int): Maximum number of clusters to consider.
        min_features (int): Minimum number of features to include in clustering.
        max_features (int): Maximum number of features to include in clustering.
        wcss_min_max (tuple): Tuple containing the minimum and maximum WCSS values.
        d_min_max (tuple): Tuple containing the minimum and maximum centroid distances.
    """

    def __init__(self, df, columns=None, min_clusters=1, max_clusters=10, min_features=1, max_features=None):
        """
        Initializes the EvoCluster object with the DataFrame and parameter specifications for clustering.

        Args:
            df (pd.DataFrame): The DataFrame to perform clustering on.
            columns (list of str, optional): Specific columns to use for clustering. Uses all columns by default.
            min_clusters (int, optional): Minimum number of clusters. Default is 1.
            max_clusters (int, optional): Maximum number of clusters. Default is 10.
            min_features (int, optional): Minimum number of features to select. Default is 1.
            max_features (int, optional): Maximum number of features to select. Defaults to the max number of columns if not specified.
        """
        self.columns = columns if columns is not None else df.columns.tolist()
        self.df = df[self.columns]
        if max_features is None:
            max_features = len(self.columns)
        validate_input(df, columns, min_clusters, max_clusters, min_features, max_features)
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_features = min_features
        self.max_features = max_features
        self.wcss_min_max, self.d_min_max = estimate_wcss_and_distance_ranges(self.df, self.columns, (self.min_clusters, self.max_clusters))

    def run(self, population_size=None, num_generations=100, mutation_rate=0.5, special_columns_indices=[]):
        """
        Runs the evolutionary algorithm to find the optimal clustering configuration.

        Args:
            population_size (int, optional): Number of configurations in each generation. Defaults to the smaller of 50 or half the square of the number of columns.
            num_generations (int, optional): Number of generations for the evolutionary process. Default is 100.
            mutation_rate (float, optional): Probability of mutating a given feature in a configuration. Default is 0.5.
            special_columns_indices (list of int, optional): Indices of columns that have constraints on how they can be selected. Defaults to an empty list.

        Returns:
            list: Top configurations across generations, ranked by fitness.
        """
        if population_size is None:
            population_size = (min(50, (len(self.columns)**2)//2))
        self.population_size = population_size
        self.special_columns_indices = special_columns_indices
        self.num_generations = num_generations
        best_configs = evolutionary_algorithm(self, self.population_size, self.num_generations, mutation_rate)
        return best_configs
