import cv2 as cv
import numpy as np
from cv2 import aruco
import pygame
from pygame.locals import *

# Constants
HEIGHT = 650
WIDTH = 650

INORGANIC_TARGET_IDS = {16, 17, 18, 19, 20}
ORGANIC_TARGET_IDS = {11, 12, 13, 14, 15}
DESTINATION_IDS = {8, 9}
BOT_IDS = {5, 6}
BOUNDARY_MARKER_IDS = {0, 2, 3, 1}

def initialize_pygame():
    """
    Inintializing pygame 
    Return: 
        pygame screen
    """
    pygame.init()
    marker_screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Marker Positions")
    return marker_screen



def aruco_detector(img,dictionaries,camera_matrix,distortion_coefficients,draw=True):
    """
    this function detect the aruco markers 
    
    Agrs:
    img: video frame
        dictionaries: aruco dictionary used
        camera_matrix: camera matrix from camera calibration
        distortion_cofficient: distortion matrix form camera calibration

    return:
        corners: a list of corners of arucos
        ids: Ids of aruco markers
    """
    dictionary=aruco.Dictionary_get(dictionaries)
    corners, ids, _ = aruco.detectMarkers(img, dictionary)
    # camera calibration
    undistort_img=cv.undistort(img, camera_matrix, distortion_coefficients)
    corners, ids, _ = aruco.detectMarkers(undistort_img, dictionary)

    if ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)

    return corners,ids

def mapping_boundary(frame,ids,corners,marker_screen,boundary_marker_centers):  
    """This function draws maps in pygame.
    Args:
    marker_screen: pygame frame
    boundary_marker_center: centre of boundary aruco markers

    Returns: none
    """  
    # running = True
    # while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # pygame_camera_frame = pygame.surfarray.make_surface(camera_frame.swapaxes(0, 1))
    pygame.display.flip()
    # Draw marker positions on the marker window
    marker_screen.fill((255, 255, 255))  # Clear the marker window
    if ids is not None:
        for i in range(len(ids)):
            marker_pos = (int(corners[i][0, 0, 0]), int(corners[i][0, 0, 1]))
            pygame.draw.circle(marker_screen, (255, 0, 0), marker_pos, 5)  # Draw a red circle at the marker position
        if len(boundary_marker_centers) == len(BOUNDARY_MARKER_IDS):
            # Create a convex hull from the marker centers
            hull = cv.convexHull(np.array(boundary_marker_centers))

            # Draw the convex hull on the camera feed
            cv.polylines(frame, [hull], isClosed=True, color=(0, 255, 0), thickness=2)

            # Create a mask for the area inside the convex hull
            mask = np.zeros_like(frame)
            cv.fillPoly(mask, [hull], (255, 255, 255))
            frame_inside_boundary = cv.bitwise_and(frame, mask)

            # Resize the mapped environment frame to match map window size
            frame_inside_boundary = cv.resize(frame_inside_boundary, (WIDTH, HEIGHT))

            # Convert the mapped environment frame to Pygame surface
            frame_inside_boundary = cv.cvtColor(frame_inside_boundary, cv.COLOR_BGR2RGB)
            pygame_map_frame = pygame.surfarray.make_surface(frame_inside_boundary.swapaxes(0, 1))

            # Blit the mapped environment frame onto the map window
            marker_screen.blit(pygame_map_frame, (0, 0))

            # Update the map display
            pygame.display.flip()
    # Update the marker display
    pygame.display.flip()
pygame.quit()


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


def main():
    cap=cv.VideoCapture(0)
    dictionaries = cv.aruco.DICT_6X6_250
                # cv.aruco.DICT_6X6_1000,
                # cv.aruco.DICT_6X6_50,
                # cv.aruco.DICT_6X6_100,
    # initializing pygame
    marker_screen = initialize_pygame()
    #  load calibration data
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
        
        # finding the centers of the aruco markerss
        boundary_markers_center=aruco_center_position(id,corners,BOUNDARY_MARKER_IDS)
        organic_marker_center=aruco_center_position(id,corners,ORGANIC_TARGET_IDS)
        inorganic_marker_center=aruco_center_position(id,corners,INORGANIC_TARGET_IDS)
        destination_marker_center=aruco_center_position(id,corners,DESTINATION_IDS)
        bot_marker_center=aruco_center_position(id,corners,BOT_IDS)

        # mapping boundary and drawing convex hall
        mapping_boundary(img_gray,id,corners,marker_screen,boundary_markers_center)

        cv.imshow("Frame",img_gray)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    # Release the camera and close all windows
    cap.release()
    cv.destroyAllWindows()
if __name__=="__main__":
    main()