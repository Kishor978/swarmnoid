import cv2 as cv
import numpy as np
from cv2 import aruco
import pygame
from pygame.locals import *

# Constants
HEIGHT = 650
WIDTH = 650

def aruco_detector(img,dictionaries,camera_matrix,distortion_coefficients,draw=True):
    dictionary=aruco.Dictionary_get(dictionaries)
    corners, ids, _ = aruco.detectMarkers(img, dictionary)
    # camera calibration
    undistort_img=cv.undistort(img, camera_matrix, distortion_coefficients)
    corners, ids, _ = aruco.detectMarkers(undistort_img, dictionary)

    if ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)

    return corners,ids


def main():
    cap=cv.VideoCapture(0)
    dictionaries = cv.aruco.DICT_6X6_250
                # cv.aruco.DICT_6X6_1000,
                # cv.aruco.DICT_6X6_50,
                # cv.aruco.DICT_6X6_100,
    calib_data_path = "E:\\swarmnoid\\calib_data\\MultiMatrix.npz"
    calib_data = np.load(calib_data_path)

    # print(calib_data.files)
    marker_size=14.9
    cam_mat = calib_data["camMatrix"]
    dist_coef = calib_data["distCoef"]
    r_vectors = calib_data["rVector"]
    t_vectors = calib_data["tVector"]

    while True:
        success,img=cap.read()
        img = cv.resize(img, (WIDTH, HEIGHT))
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # detecting markers and getting cornors
        corners,id=aruco_detector(img_gray,dictionaries,cam_mat,dist_coef,draw=True)
        
        cv.imshow("Frame",img_gray)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    # Release the camera and close all windows
    cap.release()
    cv.destroyAllWindows()
if __name__=="__main__":
    main()