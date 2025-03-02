import numpy as np
from sklearn.datasets import make_blobs
from src.cluster_estimation import cluster_estimation


def test_cluster_estimation():

    # Generate synthetic data for testing
    n_samples = 300
    n_outliers = 100
    random_state = 42

    # Create a normal cluster & outlier points
    cluster_data_points, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=0.60, random_state=random_state)
    outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))

    # Set parameters for the clustering function
    epsilon = 1.0
    radius = 1.0
    p = 1.0
    minimum_data_points = 5
    support_fraction = 0.5
    largest = True  # Change to False to test the other branch

    result = cluster_estimation(cluster_data_points, outliers, epsilon, radius, p, minimum_data_points, support_fraction, largest)
    print("Cluster Estimation Result:", result)

# Run the test
if __name__ == "__main__":
    test_cluster_estimation()
