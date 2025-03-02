import numpy as np
from sklearn.datasets import make_blobs
from src.change_detection.similarity_based_clustering import largest_cluster, mountain_method, largest_clusters
from src.change_detection.minimum_covariance_determinant import mcv_robust_clustering
from sklearn.decomposition import PCA
from scipy import stats

# Define the cluster estimation function as provided
def cluster_estimation(cluster_data_points, outliers, epsilon, radius, p, minimum_data_points, support_fraction, largest=False):

    pca_cluster = PCA(n_components=1)
    pca_outliers = PCA(n_components=1)
    principal_component_cluster = pca_cluster.fit_transform(cluster_data_points)
    principal_component_outliers = pca_outliers.fit_transform(outliers)

    cluster_flat = principal_component_cluster.flatten()
    outliers_flat = principal_component_outliers.flatten()

    alpha_c = 0.05
    ks_value, p_value = stats.ks_2samp(cluster_flat, outliers_flat)
    list_of_phi = []
    if p_value < alpha_c:
        print("Reject the null hypothesis: Create a new cluster.")
        centers = mountain_method(outliers=outliers, epsilon=epsilon, radius=radius, p=p)
        print(centers.tolist())

        if largest:
            o_tilde = largest_cluster(centers=centers.tolist(), outliers=outliers, epsilon=epsilon)
            phi = mcv_robust_clustering(cluster=o_tilde, minimum_datapoints=minimum_data_points, support_fraction=support_fraction)
            return phi
        else:
            list_of_o_tilde = largest_clusters(centers=centers, outliers=outliers, epsilon=epsilon)
            for o_tilde in list_of_o_tilde:
                phi = mcv_robust_clustering(cluster=o_tilde, minimum_datapoints=minimum_data_points, support_fraction=support_fraction)
                list_of_phi.append(phi)
            return list_of_phi

    else:
        print("Fail to reject the null hypothesis: No new cluster needed.")
        return None

# Test the cluster_estimation function
def test_cluster_estimation():
    # Generate synthetic data for testing
    n_samples = 300
    n_outliers = 100
    random_state = 42

    # Create a normal cluster
    cluster_data_points, _ = make_blobs(n_samples=n_samples, centers=1, cluster_std=0.60, random_state=random_state)
    print(cluster_data_points)
    # Create outlier points
    outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))

    # Set parameters for the clustering function
    epsilon = 1.0
    radius = 1.0
    p = 1.0
    minimum_data_points = 5
    support_fraction = 0.5
    largest = True  # Change to False to test the other branch

    # Call the cluster estimation function
    result = cluster_estimation(cluster_data_points, outliers, epsilon, radius, p, minimum_data_points, support_fraction, largest)

    # Print the output
    print("Cluster Estimation Result:", result)

# Run the test
if __name__ == "__main__":
    test_cluster_estimation()
