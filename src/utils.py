from scipy.spatial import distance
import numpy as np


def spatio_temporal_distance(datapoint_one, datapoint_two, p):
    dist = distance.euclidean(datapoint_one, datapoint_two)
    # todo: understand how to implement the temporal term: args index_one, index_two, penalization and i
    return dist/(2*p)

def omega_rmm(datapoint, center, radius, p):
    return np.exp(-spatio_temporal_distance(datapoint_one=datapoint, datapoint_two=center,
                                            p=p)/(2*radius**2))

def largest_cluster(centers, epsilon, outliers):
    sets = []
    centers_copied = centers.copy()
    centers_indices = list(range(len(centers)))

    while centers_copied:
        c_j = centers_copied.pop(0)
        c_j_index = centers_indices.pop(0)
        current_set = [c_j_index]
        centers_to_remove = []
        indices_to_remove = []

        for i, c_h in enumerate(centers_copied):
            if spatio_temporal_distance(c_j, c_h, p=1) <= 2 * epsilon:
                current_set.append(centers_indices[i])
                centers_to_remove.append(c_h)
                indices_to_remove.append(i)

        for index in sorted(indices_to_remove, reverse=True):
            centers_copied.pop(index)
            centers_indices.pop(index)

        sets.append(current_set)


    clusters = []
    for index_set in sets:
        clusters.append([outliers[idx] for idx in index_set]) 

    largest_cluster = max(clusters, key=len)

    return largest_cluster


def largest_clusters(centers, epsilon, outliers):
    sets = []
    centers_copied = centers.copy()
    centers_indices = list(range(len(centers)))

    while centers_copied:
        c_j = centers_copied.pop(0)
        c_j_index = centers_indices.pop(0)
        current_set = [c_j_index]
        centers_to_remove = []
        indices_to_remove = []

        for i, c_h in enumerate(centers_copied):
            if spatio_temporal_distance(c_j, c_h, p=1) <= 2 * epsilon:
                current_set.append(centers_indices[i])
                centers_to_remove.append(c_h)
                indices_to_remove.append(i)

        for index in sorted(indices_to_remove, reverse=True):
            centers_copied.pop(index)
            centers_indices.pop(index)

        sets.append(current_set)

    clusters = []
    for index_set in sets:
        clusters.append([outliers[idx] for idx in index_set])
    # largest_cluster = [arr.reshape((3, 1)) for arr in largest_cluster]

    return clusters