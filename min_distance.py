import numpy as np
import math

def distance_to_points(point, points):
  """
  Calculates the Euclidean distance between a point and each point in a list of points.

  Args:
    point: A list or NumPy array representing the coordinates of a single point.
    points: A list of lists or a NumPy array representing the coordinates of multiple points.

  Returns:
    A NumPy array containing the distances between the input point and each point in the list.
  """

  # Convert input to NumPy arrays for efficient calculations
  point = np.array(point)
  points = np.array(points)

  # Calculate the squared differences along each dimension
  squared_differences = (points - point)**2

  # Sum the squared differences for each point
  squared_distances = np.sum(squared_differences, axis=1)

  # Take the square root to get the Euclidean distances
  distances = np.sqrt(squared_distances)

  return distances

def find_min_distances(fixed_points, points):
    """
    Calculates the minimum distances between fixed points and points in a list,
    iteratively removing the point with the minimum distance in each cycle.

    Args:
        fixed_points: A list of lists or NumPy arrays representing the fixed points.
        points: A list of lists or NumPy arrays representing the points to compare.

    Returns:
        A list of the minimum distances found in each cycle.
    """

    min_distances = []
    min_points=[]
    for fixed_point in fixed_points:
        # Calculate distances to all remaining points
        distances = distance_to_points(fixed_point, points)

        # Find the index of the point with the minimum distance
        min_index = np.argmin(distances)

        # Print the minimum distance and corresponding point
        min_distance = distances[min_index]
        min_point = points[min_index]
        # print(f"Minimum distance from point {fixed_point}: {min_distance} to point {min_point}")

        # Add the minimum distance to the list
        min_distances.append(min_distance)
        min_points.append(min_point)
        # Remove the point with the minimum distance from the list of points
        points.pop(min_index)

    return min_distances, min_points


def angle_between_head_waste(point1, point2, reference_point):
    """
    Calculate the angle (in degrees) between two vectors formed by points in 2D space
    with respect to a reference point.

    Parameters:
    - point1 (list): A 2D list representing the coordinates of the head of bot.
    - point2 (list): A 2D list representing the coordinates of the position of waste nearest to the bot.
    - reference_point (list): A 2D list representing the coordinates of the center of the bot.

    Returns:
    - float or None: The angle in degrees between the vectors formed by point1 and point2
      with reference to the given reference_point. Returns None if the magnitude of any
      vector is zero to avoid math domain errors.
    """
    # Calculate vectors from the reference point to the other two points
    vector1 = [point1[0] - reference_point[0], point1[1] - reference_point[1]]
    vector2 = [point2[0] - reference_point[0], point2[1] - reference_point[1]]

    # Calculate the dot product of the two vectors
    dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

    # Calculate the magnitudes of the vectors
    magnitude1 = math.sqrt(sum(v**2 for v in vector1))
    magnitude2 = math.sqrt(sum(v**2 for v in vector2))

    # Check for division by zero to avoid math domain errors
    if magnitude1 == 0 or magnitude2 == 0:
        return None

    # Calculate the cosine of the angle
    cosine_angle = dot_product / (magnitude1 * magnitude2)

    # Calculate the angle in radians and convert to degrees
    angle_rad = math.acos(cosine_angle)
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def aruco_center_position(ids,corners,given_marker_ids):
    """this function finds the center of the aruco marker
    args:
        given_marker_ids: list of aruco markers id

    Returns: center of the aruco markers
    """
    center_position=[]
    output=[]
    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] in given_marker_ids:
                marker_center = np.mean(corners[i][0], axis=0, dtype=np.int32)
                center_position.append(marker_center)
                # output = [list(arr) for arr in center_position]
    return center_position

# # Example usage
# fixed_points = [[100, 200], [50, 150]]  # List of fixed points
# points = np.array([[463, 321], [285, 319], [289, 104], [400, 300], [150, 250],
#                    [300, 100], [200, 200], [450, 150], [500, 250], [350, 350]])

# # Find minimum distances in each cycle
# min_distances, min_points= find_min_distances(fixed_points, points.tolist())  # Convert back to list for popping

# print("\nMinimum distances in each cycle:")
# print(min_distances)
# print(min_points)
