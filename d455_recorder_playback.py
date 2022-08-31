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

def main():
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    print("Current PATH : ", CURRENT_PATH)

    list_color_frame = []
    list_depth_frame = []

    # Load pickle saved data
    with open(CURRENT_PATH + '/d455_data/list_color_frame.pickle', 'rb') as f:
        list_color_frame = pickle.load(f)
    with open(CURRENT_PATH + '/d455_data/list_depth_frame.pickle', 'rb') as f:
        list_depth_frame = pickle.load(f)
    
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
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            else:
                resized_color_image = color_image

            # original image
            original_image = resized_color_image.copy()

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

