from src.change_detection.similarity_based_clustering import largest_cluster, mountain_method, largest_clusters
from src.change_detection.minimum_covariance_determinant import mcv_robust_clustering
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np


def cluster_estimation(cluster_data_points, outliers, epsilon, radius, p, minimum_data_points, support_fraction,
                       largest=False):

    pca_cluster = PCA(n_components=1)
    pca_outliers = PCA(n_components=1)
    principal_component_cluster = pca_cluster.fit_transform(cluster_data_points)
    principal_component_outliers = pca_outliers.fit_transform(outliers)
    #print('Len of first PC for cluster datapoints:', len(principal_component_cluster))
    #print('type of datapoints:', type(principal_component_cluster))
    #print('shape of datapoints:', principal_component_cluster.shape)
    #print('Len of first PC for outliers:', len(principal_component_outliers))
    #print('type of outliers:', type(principal_component_outliers))
    #print('shape of outliers:', principal_component_outliers.shape)
    cluster_flat = principal_component_cluster.flatten()
    outliers_flat = principal_component_outliers.flatten()

    alpha_c = 0.05
    ks_value, p_value = stats.ks_2samp(cluster_flat, outliers_flat)
    list_of_phi = []
    if p_value < alpha_c:
        print("Reject the null hypothesis: Create a new cluster.")
        # mountain method is in common
        centers = mountain_method(outliers=outliers, epsilon=epsilon, radius=radius, p=p)

        if largest == True:
            # branch exploited for the slow drift estimation
            o_tilde = largest_cluster(centers=centers.tolist(), outliers=outliers, epsilon=epsilon)
            phi = mcv_robust_clustering(cluster=o_tilde,
                                  minimum_datapoints=minimum_data_points,
                                  support_fraction=support_fraction)
            return phi
        else:
            # branch exploited for the faulty clusters estimation
            list_of_o_tilde = largest_clusters(centers=centers, outliers=outliers, epsilon=epsilon)
            #print('len of list of o_tildes', len(list_of_o_tilde))
            for o_tilde in list_of_o_tilde:
                phi = mcv_robust_clustering(cluster=o_tilde,
                                            minimum_datapoints=minimum_data_points,
                                            support_fraction=support_fraction)
                list_of_phi.append(phi)
            return list_of_phi

    else:
        print("Fail to reject the null hypothesis: No new cluster needed.")
        return None
