import cv2 as cv
import numpy as np
from cv2 import aruco
import pygame
from pygame.locals import *

from min_distance import find_min_distances,angle_between_head_waste,aruco_center_position
from astar_algorithm import Pathfinding
from marker_orientation import bot_head,getMarkerOrientation

# Constants
HEIGHT = 650
WIDTH = 650
MARKER_SIZE=14.9

INORGANIC_TARGET_IDS = {16, 17, 18, 19, 20}
ORGANIC_TARGET_IDS = {11, 12, 13, 14, 15}
DESTINATION_IDS = {8, 9}
BOT_IDS = {5, 6}
BOUNDARY_MARKER_IDS = {0, 2, 3, 1}

def load_calibration_data(calib_data_path):
    calib_data = np.load(calib_data_path)
    camera_matrix = calib_data["camMatrix"]
    camera_distortion = calib_data["distCoef"]
    r_vectors = calib_data["rVector"]
    t_vectors = calib_data["tVector"]


    return camera_matrix, camera_distortion

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
# pygame.quit()

def find_and_visualize_path(start_point, end_point):
    """
    Finds the shortest path between the given start and end points using the A* algorithm,
    and visualizes the path on a Pygame window.

    Parameters:
        - start_point (list): The starting point coordinates [x, y].
        - end_point (list): The ending point coordinates [x, y].
    """

    # Create an instance of the Pathfinding class
    pathfinding = Pathfinding(WIDTH, HEIGHT)

    # Create nodes
    pathfinding.create_nodes()

    # Find the closest nodes to the starting and ending points
    start_node = pathfinding.nodes[start_point[0] // pathfinding.pixel_size][start_point[1] // pathfinding.pixel_size]
    end_node = pathfinding.nodes[end_point[0] // pathfinding.pixel_size][end_point[1] // pathfinding.pixel_size]

    # Find the path passing through the closest sub-destination
    path = pathfinding.astar(start_node, end_node)

    # Draw grid, sub-destinations, and path
    pathfinding.draw_path(path)

    # Move ball along the path and visualize
    pathfinding.move_ball_along_path(path)

    # pygame.quit()

def main():
    cap=cv.VideoCapture(0)
    dictionaries = cv.aruco.DICT_6X6_250

    # initializing pygame
    marker_screen = initialize_pygame()

    #  load calibration data
    calib_data_path = "E:\\swarmnoid\\calib_data\\MultiMatrix.npz"
    cam_mat,dist_coef=load_calibration_data(calib_data_path)
    
    
    while True:
        success,img=cap.read()
        img = cv.resize(img, (WIDTH, HEIGHT))
        img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # detecting markers and getting cornors
        corners,id=aruco_detector(img_gray,dictionaries,cam_mat,dist_coef,draw=True)
        # print(corners)

        # finding the centers of the aruco markerss
        boundary_markers_center=aruco_center_position(id,corners,BOUNDARY_MARKER_IDS)
        organic_marker_center=aruco_center_position(id,corners,ORGANIC_TARGET_IDS)
        inorganic_marker_center=aruco_center_position(id,corners,INORGANIC_TARGET_IDS)
        destination_marker_center=aruco_center_position(id,corners,DESTINATION_IDS)
        bot_marker_center=aruco_center_position(id,corners,BOT_IDS)

        # Finding the position of head of the point in x axis
        head_of_bot=bot_head(id,corners,BOT_IDS)
        # print(head_of_bot)

        # mapping boundary and drawing convex hall
        mapping_boundary(img_gray,id,corners,marker_screen,boundary_markers_center)

        # merging lists to get list of position of all waste_aruco marker
        # Merge the lists
        merged_list = organic_marker_center.copy()  # Create a copy of A to avoid modifying the original list
        merged_list.extend(inorganic_marker_center)
        # print(merged_list)
        
        if len(merged_list)>3 and len(bot_marker_center)>0:
             # getting the closest waste position
            _,min_point=find_min_distances(bot_marker_center,merged_list)
            _,min_point1=find_min_distances(bot_marker_center,merged_list)
            # print("m",min_point)
            # print("n",min_point1)

            # getting the orientation of bots
            x,y,z,roll,pitch,yam=getMarkerOrientation(corners,MARKER_SIZE)
            start=bot_marker_center[0]
            end=min_point[0]
            end1=min_point1[0]

            # getting the angle between bot_head and waste
            angle=angle_between_head_waste(head_of_bot[0],end,start)
            # print("angle",angle)

            # print("ap",start)
            # print("pa",end)
            # pygame.time.delay(5000) 
            # find_and_visualize_path(start,end)
            # find_and_visualize_path(start,end1)
        
        cv.imshow("Frame",img_gray)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    # Release the camera and close all windows
    cap.release()
    cv.destroyAllWindows()
if __name__=="__main__":
    main()