import numpy as np
from sklearn.linear_model import LinearRegression

def is_straight_line(points, tolerance=0.01):
    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    residuals = np.abs(predictions - y)
    return np.all(residuals < tolerance)

def regularize_straight_line(points):
    if is_straight_line(points):
        return np.array([points[0], points[-1]])
    return points

def is_circle(points, tolerance=0.01):
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return np.all(np.abs(distances - np.mean(distances)) < tolerance)

def regularize_circle(points):
    if is_circle(points):
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        num_points = 100
        angles = np.linspace(0, 2 * np.pi, num_points)
        circle_points = np.array([[center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)] for angle in angles])
        return circle_points
    return points

def is_rectangle(points, tolerance=0.01):
    if len(points) != 4:
        return False
    distances = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    diagonals = np.linalg.norm(np.roll(points, -2, axis=0) - points, axis=1)
    return np.allclose(distances[:2], distances[2:], atol=tolerance) and np.allclose(diagonals, np.sqrt(2) * distances[0], atol=tolerance)

def regularize_rectangle(points):
    if is_rectangle(points):
        center = np.mean(points, axis=0)
        widths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        width, height = np.mean(widths[:2]), np.mean(widths[2:])
        half_width, half_height = width / 2, height / 2
        rectangle_points = np.array([
            [center[0] - half_width, center[1] - half_height],
            [center[0] + half_width, center[1] - half_height],
            [center[0] + half_width, center[1] + half_height],
            [center[0] - half_width, center[1] + half_height]
        ])
        return rectangle_points
    return points

def is_regular_polygon(points, tolerance=0.01):
    num_points = len(points)
    if num_points < 3:
        return False
    distances = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    angles = np.arccos(np.clip(np.sum(np.roll(points, -1, axis=0) * points, axis=1) / (np.linalg.norm(np.roll(points, -1, axis=0), axis=1) * np.linalg.norm(points, axis=1)), -1.0, 1.0))
    return np.allclose(distances, np.mean(distances), atol=tolerance) and np.allclose(angles, np.mean(angles), atol=tolerance)

def regularize_regular_polygon(points):
    if is_regular_polygon(points):
        center = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - center, axis=1))
        num_points = len(points)
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        polygon_points = np.array([[center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)] for angle in angles])
        return polygon_points
    return points
