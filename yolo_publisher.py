#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

if __name__ == '__main__':
   rospy.init_node("camera_publisher", anonymous=True) #anonymous는 name conflictf를 피하기 위해 node의 이름을 바꿔주는 것
   pub = rospy.Publisher("video_topic", Image, queue_size=60)  #토픽 이름은 subscriber와 같게 해야 한다.
   rate = rospy.Rate(100)


   webcam = cv2.VideoCapture(4)
   bridge = CvBridge()

   model = YOLO('/home/yongjin/catkin_ws/src/opencv_ros/scripts/yolo_obstacle_seg_new.pt')
   yolo_classes = list(model.names.values())
   classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
   colors = [[0, 0 ,255], [0,255,0]]
   conf = 0.5

   while not rospy.is_shutdown():
       status, img = webcam.read()
       if status == True:
           rospy.loginfo("Video frame captured and published")

           results = model.predict(img, conf=conf)
           if(results[0].masks != None):  #찾은 것이 있으면 색을 칠한다.
               for result in results:
                  for mask, box in zip(result.masks.xy, result.boxes):
                    points = np.int32([mask])
                    color_number = classes_ids.index(int(box.cls[0]))
                    cv2.fillPoly(img, points, colors[color_number])
               
           TransmitImage = bridge.cv2_to_imgmsg(img) #opencv image에서 ros image로 변환
           pub.publish(TransmitImage)
           rate.sleep()
           
