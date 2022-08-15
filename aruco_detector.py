import pyrealsense2 as rs
import argparse
import imutils
import time
import cv2
import sys
import numpy as np
import glob
import os

SHOW_DEPTH = False

IMAGE_RGB_PATH = "./images/image_{:01d}.jpg"
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
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT['DICT_ARUCO_ORIGINAL'])
    arucoParams = cv2.aruco.DetectorParameters_create()
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    
    frameId = 0
    print("Current PATH : ", CURRENT_PATH)

    if glob.glob(CURRENT_PATH + "/image_counter.txt"):
        print("File exist")
        with open(CURRENT_PATH + "/image_counter.txt", "r") as files:
            lines = files.readlines()
            if len(lines) > 0:
                text = lines[0]
                frameId = int(text)
    else:
        print("File not exist")
        with open(CURRENT_PATH + "/image_counter.txt", "w") as files:
            files.writelines(str(frameId))

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # print("Distance at x=320, y=240 : ", depth_frame.get_distance(320,240))
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
            # create zero image
            roi = np.zeros((300,300,3), np.uint8)

            # detect ArUco markers in the input frame
            (corners, ids, rejected) = cv2.aruco.detectMarkers(color_image, arucoDict, parameters=arucoParams)
            # verify *at least* one ArUco marker was detected
            if len(corners) == 4:
                # flatten the ArUco IDs list
                ids = ids.flatten()
                # loop over the detected ArUCo corners
                list_marker_points = [[0,0],[0,0],[0,0],[0,0],[0,0]] # order dalam list 3, 2, 1, 0
                
                print("======")
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
                                       
                    print("Distance at Marker "+ str(markerID) + ":" + str(depth_frame.get_distance(cX, cY)))
                    cv2.circle(resized_color_image, (cX, cY), 4, (0, 0, 255), -1)
                    # draw the ArUco marker ID on the frame
                    cv2.putText(resized_color_image, str(markerID), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                initial_points = np.float32([list_marker_points[0],list_marker_points[1],list_marker_points[2],list_marker_points[3]])
                target_points = np.float32([[0,300], [0,0], [300,300], [300,0]])
                M = cv2.getPerspectiveTransform(initial_points, target_points)
                dst = cv2.warpPerspective(original_image, M, (300,300))
                roi = cv2.rotate(dst.copy(), cv2.ROTATE_90_CLOCKWISE)
            
            if SHOW_DEPTH:
                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))
            else:
                images = resized_color_image
            
            cv2.namedWindow("RealSense Viewer", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense Viewer", images)
            cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Result", roi)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            elif key == ord('s'):
                cv2.imwrite(IMAGE_RGB_PATH.format(frameId), roi)
                print ('Capture %d'%(frameId))
                frameId += 1
                with open(CURRENT_PATH + "/image_counter.txt", "w") as files:
                    files.writelines(str(frameId))
    finally:
        # Stop streaming
        pipeline.stop()
        # do a bit of cleanup
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    main()

