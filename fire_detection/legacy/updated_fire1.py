#!/usr/bin/python3

import csv
import cv2
import numpy as np
import rospy
from std_msgs.msg import Int16

class FireDetectionNode:
    def __init__(self):
        rospy.init_node('fire_detection_node')
        
        # Parameters for object detection
        self.known_distance = 182.88  # distance in cm
        self.known_width = 22.86      # width in cm
        self.focal_length = 707.5     # focal length in cm
        self.lower = np.array([14, 90, 90], dtype=np.uint8)
        self.upper = np.array([30, 255, 255], dtype=np.uint8)
        self.cap = cv2.VideoCapture(0)
        self.pub = rospy.Publisher("object_detector_data", Int16, queue_size=1)
        self.rate = rospy.Rate(10)

    def distance_finder(self, known_width, focal_length, width_in_frame):
        return (known_width * focal_length) / width_in_frame

    def callback(self, data):
        rospy.loginfo(data.data)
        # field names
        fields = ['raw_data']
        # name of csv file
        filename = "object_detector_records.csv"
        # writing to csv file
        with open(filename, 'w') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            # writing headers (field names)
            writer.writeheader()
            # writing data rows
            writer.writerow({'raw_data': data.data})

    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (700, 500))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)
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
                distance = self.distance_finder(self.known_width, self.focal_length, w)
                cv2.putText(frame, "Distance: " + str(int(distance)) + "cm", (20, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
                rospy.loginfo(f"Object width in pixels: {w}")
                self.pub.publish(Int16(data=w))

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.rate.sleep()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = FireDetectionNode()
    node.run()
