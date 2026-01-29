#!/usr/bin/env python3
import rospy
from scout_msgs.msg import ScoutStatus
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import os
import json
import cv2
import base64
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import json
import base64
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset
import cv2
import glob
from cil_net import CIL_net

class DataSynchronizer:
    def __init__(self):
        rospy.init_node('data_synchronizer')
        
        self.scout_sub = rospy.Subscriber("/scout_status", ScoutStatus, self.scout_callback)
        self.scout_data = None

        self.camera_sub = rospy.Subscriber("video_topic", Image, self.camera_callback)
        self.camera_image = None
        self.bridge = CvBridge()
        self.model = CIL_net()


        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.scout_freq = 10.0
        self.camera_freq = 10.0
        self.scout_rate = rospy.Rate(self.scout_freq)
        self.camera_rate = rospy.Rate(self.camera_freq)
        
    def scout_callback(self, data):
        self.scout_data = data
        
    def camera_callback(self, data):
        self.camera_image = data

    def calculate_velocity(self):
        msg = Twist()
        self.model.load_state_dict(torch.load('/home/yongjin/venv/torch_venv/scripts/CIL/best_turn_2.pt'))
        self.model.eval()
        prev_linear_velocity = 0.3
        rospy.loginfo("Node has been started.")
        while not rospy.is_shutdown():
            scout_data = self.scout_data
            linear_velocity = scout_data.linear_velocity

            camera_image = self.camera_image
            image_cv = self.bridge.imgmsg_to_cv2(camera_image)
            image_cv = cv2.resize(image_cv, (200, 88))
            
            image_np = np.array(image_cv)
            with torch.no_grad():
                # Convert image to torch tensor if needed
                image_tensor = torch.from_numpy(image_np.transpose((2, 0, 1))).float().div(255.0).unsqueeze(0)
                linear_velocity = torch.tensor(linear_velocity)
                mode = torch.tensor(scout_data.front_light_state.mode)
                pred_control, _ = self.model(image_tensor, linear_velocity)

            #velocity 계산
            acceleration = pred_control[0, 0]
            curr_lin_vel = prev_linear_velocity + acceleration * (1.0 / self.scout_freq)
            if curr_lin_vel < 0:
                curr_lin_vel = 0
            angular_velocity = pred_control[0, 1]
            print("linear_velocity <-", curr_lin_vel)
            print("angular_velocity <-", angular_velocity)
            print("=====================================")
            msg.linear.x = curr_lin_vel
            #msg.linear.x = 0.25
            msg.angular.z = angular_velocity
            self.pub.publish(msg)

            prev_linear_velocity = curr_lin_vel
            self.scout_rate.sleep()
            self.camera_rate.sleep()

if __name__ == '__main__':
    try:
        synchronizer = DataSynchronizer()
        synchronizer.calculate_velocity()
    except rospy.ROSInterruptException:
        pass



