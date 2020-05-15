import numpy as np


def generate_random_point_nd(num_points: int = 2, n: int = 10, min_: int = 0, max_: int = 10) -> np.array:
    """
    Generates n random points in hypercube [xy_min; xy_max]
    :param num_points: int,
    :param n: int,
    :param min_: int,
    :param max_: int,
    :return: np.array,
    """
    xy_min = [min_] * n
    xy_max = [max_] * n
    data = np.random.uniform(low=xy_min, high=xy_max, size=(num_points, n))
    return data


def distance(x1: np.array, x2: np.array) -> float:
    """
    Calculates Eucledian distance between two points
    :param x1: np.array, point one
    :param x2: np.array, point two
    :return: float, Euclidian distance between x1 and x2
    """
    return np.linalg.norm(x1 - x2)


def generate_random_point(n: int = 2, xy_min: list = [0, 0], xy_max: list = [10, 20]) -> np.array:
    """
    Generates random point in rectangle [xy_min; xy_max]
    :param n: int, number of generated points
    :param xy_min: list, left bottom point
    :param xy_max: list, right top point
    :return: np.array, generated points
    """
    data = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
    return data
