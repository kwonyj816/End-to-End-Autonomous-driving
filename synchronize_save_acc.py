#!/usr/bin/env python3
import rospy
from scout_msgs.msg import ScoutStatus
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import json
import cv2
import base64

class DataSynchronizer:
    def __init__(self):
        rospy.init_node('data_synchronizer')
        
        self.scout_sub = rospy.Subscriber("/scout_status", ScoutStatus, self.scout_callback)
        self.scout_data = None

        self.camera_sub = rospy.Subscriber("video_topic", Image, self.camera_callback)
        self.camera_image = None
        self.bridge = CvBridge()

        self.data_pairs = []

        self.data_directory = rospy.get_param("~data_directory_turn", "data_directory_turn")
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

        self.scout_freq = 10.0
        self.camera_freq = 10.0
        self.scout_rate = rospy.Rate(self.scout_freq)
        self.camera_rate = rospy.Rate(self.camera_freq)

    def scout_callback(self, data):
        self.scout_data = data
        
    def camera_callback(self, data):
        self.camera_image = data

    def process_and_save_data(self):
        flag = False
        prev_linear_velocity = None
        prev_angular_velocity = None
        prev_mode = None
        prev_image = None

        while not rospy.is_shutdown():
            scout_data = self.scout_data
            camera_image = self.camera_image

            if scout_data is not None and camera_image is not None:
                cv_image = self.bridge.imgmsg_to_cv2(camera_image)
                cv_image = cv2.resize(cv_image, (200, 88))

                _, buffer = cv2.imencode('.jpg', cv_image)
                image = base64.b64encode(buffer).decode('utf-8')
                
                if flag == True:
                    acceleration = (scout_data.linear_velocity - prev_linear_velocity) / (1.0/self.scout_freq)
                else:
                    acceleration = None

                data_pair = {
                    'status':{
                        'linear_velocity': prev_linear_velocity,
                        'angular_velocity': prev_angular_velocity,
                        'acceleration' : acceleration
                    },
                    'mode' : prev_mode,
                    'image': prev_image
                }
                if flag == True:
                    self.data_pairs.append(data_pair)
                    rospy.loginfo("Saved data pair. Total pairs: %d", len(self.data_pairs))

                    filename = os.path.join(self.data_directory,
                                             "data_{}.json".format(rospy.get_rostime()))    #str 추가
                    with open(filename, 'w') as file:
                        json.dump(data_pair, file)
                else:
                    flag = True
                
                prev_linear_velocity = scout_data.linear_velocity
                prev_angular_velocity = scout_data.angular_velocity
                prev_mode = scout_data.front_light_state.mode
                prev_image = image

            self.scout_rate.sleep()
            self.camera_rate.sleep()

if __name__ == '__main__':
    try:
        synchronizer = DataSynchronizer()
        synchronizer.process_and_save_data()
    except rospy.ROSInterruptException:
        pass
