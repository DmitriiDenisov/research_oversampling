import numpy as np


def test_points_on_line_old(p1, p2, list_p3, tol=10):
    # Calculate two vectors and check if they are collinear
    if len(list_p3.shape) == 1:
        list_p3 = list_p3[np.newaxis, :]
    vec_1 = p2 - p1
    for p3 in list_p3:
        vec_2 = (p3 - p1) / vec_1
        # take unique values
        unique_arr = np.unique(vec_2[~np.isnan(vec_2)].round(decimals=tol))
        if len(unique_arr) > 1:
            return False
        # Check that if nan then there were two zeros
        idx = np.where(~np.isfinite(vec_2))[0]
        if idx.size:
            if not (all((p3 - p1)[idx] == vec_1[idx]) and all(vec_1[idx] == 0)):
                return False
        return True
    return True


def test_points_on_line(p1, p2, list_p3, tol=0.000000000001):
    # Calculate two vectors and check if they are collinear
    if len(list_p3.shape) == 1:
        list_p3 = list_p3[np.newaxis, :]
    vec_1 = p2 - p1
    for p3 in list_p3:
        vec_2 = p3 - p1

        if np.linalg.norm(vec_2) > tol and np.linalg.norm(vec_1) > tol:
            if abs(np.dot(vec_2, vec_1)) / (np.linalg.norm(vec_2) * np.linalg.norm(vec_1)) < 1 - tol:
                print(np.linalg.norm(vec_2))
                print(np.linalg.norm(vec_1))
                return False
    return True
