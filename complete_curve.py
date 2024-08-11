import numpy as np
from scipy.interpolate import splprep, splev

def complete_curve(points):
    tck, u = splprep(points.T, s=0)
    new_points = splev(np.linspace(0, 1, len(points) * 2), tck)
    return np.vstack(new_points).T
