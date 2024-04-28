# EvoCluster: Evolutionary Clustering Tool

EvoCluster effectively uses an evolutionary algorithm to optimize both feature selection and the number of clusters for KMeans clustering.

Given a dataset with a large number of possible features, users can call Evo_Cluster and find the optimal: Features, number of features, number of clusters.

This project aims to automate the process of identifying the optimal clustering configuration by evaluating silhouette scores, within-cluster sum of squares (WCSS), 
and distances between cluster centroids.

Particularly useful when dealing with a dataset with a large number of features, eg: 50+

## Features

- **Feature Selection**: Dynamically selects the optimal set of features for clustering.
- **Cluster Optimization**: Determines the ideal number of clusters.
- **Evaluation Metrics**: Utilizes silhouette score, WCSS, and centroid distances to evaluate clustering configurations.
- **Evolutionary Algorithm**: Employs genetic operations including mutation, tournament selection and crossover to evolve clustering configurations over generations.

## Installation

To install EvoCluster, you will need Python 3.6 or later. Clone this repository and install the required packages using pip:

```bash
git clone https://github.com/joshkatt/Evo-Cluster.git
cd evo-cluster
pip install -r requirements.txt
```
## Usage
Here is a quick example to get you started with EvoCluster:
```
from evo_cluster import EvoCluster
import pandas as pd

# Load your dataset
data = pd.read_csv('your_data.csv')

# Initialize EvoCluster
cluster_tool = EvoCluster(df=data, columns=['feature1', 'feature2'])

# Run the evolutionary clustering
best_configurations = cluster_tool.run()

print("Top configurations:", best_configurations)

```

You can specify the minimum/maximum number of features to use or a range for the number of clusters at run time to fit Evo_Cluster to your requirements.

You can even spcify special rules if many features represent correlated data, for example you can specify that only a maximum of 1 out of 4 special columns can be considered at a time.

## Modules
core.py: Contains the EvoCluster class which is the main interface for clustering.
algo.py: Implements the evolutionary algorithm, fitness calculations, and genetic operators.
cluster.py: Provides functionality to perform clustering and compute metrics.

## Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Distributed under the MIT License. See LICENSE for more information.

## Authors
Joshua Kattapuram - joshuakatt
email: joshuakattapuram10@gmail.com
