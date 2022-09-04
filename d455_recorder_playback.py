import argparse
import imutils
import time
import cv2
import sys
import numpy as np
import glob
import os
import pickle

SHOW_DEPTH = False
CURRENT_PATH = os.getcwd()

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def main():
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    print("Current PATH : ", CURRENT_PATH)

    list_color_frame = []
    list_depth_frame = []

    # Load pickle saved data
    with open(CURRENT_PATH + '/d455_data/list_color_frame.pickle.5b', 'rb') as f:
        list_color_frame = pickle.load(f)
    with open(CURRENT_PATH + '/d455_data/list_depth_frame.pickle.5b', 'rb') as f:
        list_depth_frame = pickle.load(f)
    
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT['DICT_ARUCO_ORIGINAL'])
    arucoParams = cv2.aruco.DetectorParameters_create()

    print("list :", list_color_frame)

    counter = 0

    try:
        while True:
            # Load image from list
            depth_image = list_depth_frame[counter]
            color_image = list_color_frame[counter]

            # print("Distance at x=320, y=240 : ", depth_frame.get_distance(320,240))
            # print("Depth at x=320, y=240 : ", depth_image[240, 320])

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape           
            
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image.copy(), dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            else:
                resized_color_image = color_image.copy()

            # # original image
            # original_image = resized_color_image.copy()

            (corners, ids, rejected) = cv2.aruco.detectMarkers(color_image.copy(), arucoDict, parameters=arucoParams)

            if len(corners) == 4:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners
                list_marker_points = [[0,0],[0,0],[0,0],[0,0],[0,0]] # order dalam list 3, 2, 1, 0

                for (markerCorner, markerID) in zip(corners, ids):
                    # extract the marker corners (which are always returned
                    # in top-left, top-right, bottom-right, and bottom-left order
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    # convert each of the (x, y)-coordinate pairs to integers
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    # draw the bounding box of the ArUCo detection
                    cv2.line(resized_color_image, topLeft, topRight, (0, 255, 0), 2)
                    cv2.line(resized_color_image, topRight, bottomRight, (0, 255, 0), 2)
                    cv2.line(resized_color_image, bottomRight, bottomLeft, (0, 255, 0), 2)
                    cv2.line(resized_color_image, bottomLeft, topLeft, (0, 255, 0), 2)
                    # compute and draw the center (x, y)-coordinates of the
                    # ArUco marker
                    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

                    if markerID < len(list_marker_points):
                        list_marker_points[markerID]=[cX,cY]

                    cv2.circle(resized_color_image, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the frame
                    cv2.putText(resized_color_image, str(markerID), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if SHOW_DEPTH:
                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
            else:
                images = resized_color_image
            
            cv2.namedWindow("RealSense RGB Viewer", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense RGB Viewer", images)
            cv2.namedWindow("RealSense Depth Viewer", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense Depth Viewer", depth_colormap)
            
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            print("Image number ", counter + 1)
            if key == ord("q"):
                break
            elif key == ord('n'):
                counter += 1
                if counter >= len(list_color_frame):
                    counter = 0
                print("Go to next image ", counter)
            elif key == ord('p'):
                counter -= 1
                if counter < 0:
                    counter = len(list_color_frame) - 1
                print("Go to previous image ", counter)
    
    finally:
        # do a bit of cleanup
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()

