
import cv2
from cv2 import aruco
import numpy as np
import networkx as nx

# # Define the dictionaries to try
dictionaries_to_try = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_4X4_1000,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_5X5_100,
    cv2.aruco.DICT_5X5_250,
    cv2.aruco.DICT_5X5_1000,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_6X6_100,
    cv2.aruco.DICT_6X6_250,
    cv2.aruco.DICT_6X6_1000,
    cv2.aruco.DICT_7X7_50,
    cv2.aruco.DICT_7X7_100,
    cv2.aruco.DICT_7X7_250,
    cv2.aruco.DICT_7X7_1000,
    cv2.aruco.DICT_ARUCO_ORIGINAL,
    # April Tag
    cv2.aruco.DICT_APRILTAG_16h5,
    cv2.aruco.DICT_APRILTAG_25h9,
    cv2.aruco.DICT_APRILTAG_36h10,
    cv2.aruco.DICT_APRILTAG_36h11,
]

# dictionaries_to_try = [
#     cv2.aruco.DICT_6X6_50,
#     cv2.aruco.DICT_6X6_100,
#     cv2.aruco.DICT_6X6_250,
#     cv2.aruco.DICT_6X6_1000,
# ]
# calib_data_path="../calib_data/MultiMatrix.npz"
# calib_data=np.load(calib_data_path)
# Initialize the camera
cap = cv2.VideoCapture(0)

# Define marker IDs for pathfinding
start_id = 0
end_id = 2
# Define boundary marker IDs
boundary_marker_ids = {0,2,3,1}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try detecting markers from different dictionaries and sizes
    for dictionary_id in dictionaries_to_try:
        # Load the predefined dictionary for ArUco markers
        dictionary = aruco.getPredefinedDictionary(dictionary_id)
        # Create an ArUco marker board
        # board = aruco.CharucoBoard((3, 3), 0.04, 0.01, dictionary)
        
        # Detect ArUco markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Update the graph with marker connections
            if (len(ids) >= 3):
                marker_centers=[]
                for i in range(len(ids)):
                    if ids[i][0] in boundary_marker_ids:
                        marker_center = np.mean(corners[i][0], axis=0, dtype=np.int32)
                        marker_centers.append(marker_center)
                        print(marker_centers)
                if len(marker_centers) == len(boundary_marker_ids):
                        hull = cv2.convexHull(np.array(marker_centers))
                        cv2.polylines(frame, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                print("Less than three markers detected. Not updating the graph.")

    # Display the frame
    cv2.imshow("ArUco Marker Detection", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
