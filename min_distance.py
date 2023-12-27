import numpy as np

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
        print(f"Minimum distance from point {fixed_point}: {min_distance} to point {min_point}")

        # Add the minimum distance to the list
        min_distances.append(min_distance)
        min_points.append(min_point)
        # Remove the point with the minimum distance from the list of points
        points.pop(min_index)

    return min_distances, min_points

# # Example usage
# fixed_points = [[100, 200], [50, 150]]  # List of fixed points
# points = np.array([[463, 321], [285, 319], [289, 104], [400, 300], [150, 250],
#                    [300, 100], [200, 200], [450, 150], [500, 250], [350, 350]])

# # Find minimum distances in each cycle
# min_distances, min_points= find_min_distances(fixed_points, points.tolist())  # Convert back to list for popping

# print("\nMinimum distances in each cycle:")
# print(min_distances)
# print(min_points)