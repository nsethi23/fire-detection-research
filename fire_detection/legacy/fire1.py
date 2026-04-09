#!/usr/bin/python3 

import csv
import cv2
import numpy as np 
from matplotlib import pyplot as plt
import random
import time
from enum import Enum
import rospy
from std_msgs.msg import Int16


# Object Detection with distance data.
########################################################################
#yellow = [0, 255, 255]
known_distance = 182.88 # distance in cm
known_width = 22.86 # width in cm
focal_length = 707.5 #focal length in cm

Green = (0, 255, 0)
RED = (0,0,255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX


cap = cv2.VideoCapture(0)


def distance_finder(known_width, focal_length, width_in_frame):
    distance = (known_width * focal_length) / width_in_frame
    return distance


lower = np.array([14,90, 90], dtype=np.uint8)
upper = np.array([30, 255, 255], dtype=np.uint8)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (700, 500))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    output = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 400:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        distance = distance_finder(known_width, focal_length, w)
        cv2.putText(frame, "Distance: " + str(int(distance)) + "cm", (20,30), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0))
        print("Object width in pixels: " + str(w))

    
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
########################################################################

# Sensor Data Callback, writes data to CSV file
def callback (data):
    # prints to terminal window
    rospy.loginfo(data)

    # field names
    fields = ['raw_data']

    # name of csv file
    filename = "object_detecter_records.csv"

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(data)
    

# ROS Subscription and Publisher 
def ROSSubscribe():
    rospy.init_node('object_detecter_node')
    rospy.Subscriber("object_detecter_data", Int16, callback)

    # queue_size???
    global pub
    pub = rospy.Publisher("object_detecter_data", Int16, queue_size=1)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rate.sleep()

# Main
if __name__ == "__main__":
    ROSSubscribe()

# Cleanup
cap.release()
cv2.destroyAllWindows()
