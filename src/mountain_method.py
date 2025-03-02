from scipy.spatial import distance
import numpy as np
import math
from src.utils import omega_rmm


def mountain_method(outliers, epsilon, radius, p):
    err = math.inf
    centers = outliers.copy()
    while err >= epsilon:
        # assuming each outlier-point is a candidate center -> optimality property
        centers_old = centers.copy()
        for i, c_j in enumerate(centers):

            num = sum(np.array(theta_h) * omega_rmm(datapoint=theta_h, center=c_j,
                                                    p=p, radius=radius) for theta_h in outliers)
            den = sum(omega_rmm(datapoint=theta_h, center=c_j,
                                p=p, radius=radius) for theta_h in outliers)
            c_j = num / den
            centers[i] = c_j
            zipped = zip(centers, centers_old)

        err = max(distance.euclidean(c_j, c_j_old) for c_j, c_j_old in zipped)

    return centers


