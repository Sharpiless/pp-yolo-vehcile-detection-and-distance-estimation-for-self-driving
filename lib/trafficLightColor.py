import cv2
import numpy as np
import argparse

def estimate_label(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # RED: lower mask (0-10)
    lower_red = np.array([0,0,0])
    upper_red = np.array([10,255,255])
    mask_red0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # RED: upper mask (170-180)
    lower_red = np.array([160,0,0])
    upper_red = np.array([180,255,255])
    mask_red1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join RED masks
    mask_red = mask_red0+mask_red1

    # YELLOW: mask (12-36)
    lower_yellow = np.array([10,0,0])
    upper_yellow = np.array([32,255,255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # GREEN: mask (45-95)
    lower_green = np.array([45,0,0])
    upper_green = np.array([95,255,255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    area_red = cv2.countNonZero(mask_red)
    area_yellow = cv2.countNonZero(mask_yellow)
    area_green = cv2.countNonZero(mask_green)

    color_map = ["RED", "YELLOW", "GREEN"]
    detected_color = color_map[np.argmax([area_red, area_yellow, area_green])]

    return detected_color