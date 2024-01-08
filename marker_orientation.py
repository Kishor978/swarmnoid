# importing required modules
import cv2
import cv2.aruco as aruco
import numpy as np
import math

# getting the camera calibration path

calib_data_path = "E:\\swarmnoid\\calib_data\\MultiMatrix.npz"
calib_data = np.load(calib_data_path)

# print(calib_data.files)
marker_size=14.9
camera_matrix = calib_data["camMatrix"]
camera_distortion = calib_data["distCoef"]
# creating 180 deg rotation matrix around the x axis
R_flip = np.zeros((3, 3), dtype=np.float32)
R_flip[0, 0] = 1.0
R_flip[1, 1] = -1.0
R_flip[2, 2] = -1.0

# function to check if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    """ Checks if the input matrix is a valid rotation matrix.

        Parameters:
        R (numpy.ndarray): Input matrix to check.
        Returns:
        bool: True if the matrix is a valid rotation matrix, False otherwise.
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)      
    return n < 1e-6

# function to calculate rotation matrix to euler angles
def rotationMatrixToEulerAngles(R):
    """
    Converts a rotation matrix to Euler angles (roll, pitch, yaw).

        Parameters:
        R (numpy.ndarray): Rotation matrix.
        Returns:
        numpy.ndarray: Euler angles (roll, pitch, yaw).
    """
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# function to compute marker attitude/orientation
def getMarkerOrientation(corners, real_marker_size):
    """
    Computes marker attitude/orientation based on detected corners.

    Parameters:
    corners (numpy.ndarray): Detected corners of the marker.
    real_marker_size (float): Actual size of the marker.
    camera_matrix (numpy.ndarray): Camera intrinsic matrix.
    camera_distortion (numpy.ndarray): Camera distortion coefficients.
    
    Returns:
    (xm, ym, zm, roll_marker, pitch_marker, yaw_marker): Marker position and orientation.
    """
    # np_corners = np.array([[[corners[0]["x"], corners[0]["y"]],
    #                         [corners[1]["x"], corners[1]["y"]],
    #                         [corners[2]["x"], corners[2]["y"]],
    #                         [corners[3]["x"], corners[3]["y"]],
    #                         ]])
    # pose estimation
    markerPose = aruco.estimatePoseSingleMarkers(corners, real_marker_size, camera_matrix, camera_distortion)
    rvec, tvec = markerPose[0][0, 0, :], markerPose[1][0, 0, :]
    # getting the rotation matrix tag -> camera
    R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
    R_tc = R_ct.T
    
    # getting the position from tvec
    xm = tvec[0]
    ym = tvec[1]
    zm = tvec[2]
    # getting the attitude from rvec (in terms of euler 321 angles)
    roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip * R_tc)
    # printing the marker orientation in terms of tvec (xm, ym, zm) and rvec (roll, pitch, yaw)
    str_position = "Marker pos [cm] x=%4.0f  y=%4.0f  z=%4.0f" % (xm, ym, zm)
    str_position += " roll=%4.0f  pitch=%4.0f  yaw=%4.0f" % (math.degrees(roll_marker), math.degrees(pitch_marker), math.degrees(yaw_marker))
    print(str_position)

    return (xm,ym,zm, roll_marker, pitch_marker, yaw_marker)


def bot_head(ids,corners,given_marker_ids):
    """this function define the head of the bots
    Returns:head of the bots
    """
    head=[]
    
    if ids is not None:
        for i in range(len(ids)):
            if ids[i][0] in given_marker_ids:
                # print(ids[i][0],corners[i][0])
                head_bot=[((corners[i][0][0][0]+corners[i][0][1][0])/2),((corners[i][0][0][1]+corners[i][0][1][1])/2)]
                head.append(head_bot)
    return head
