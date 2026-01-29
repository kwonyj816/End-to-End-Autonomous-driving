# End-to-End-Autonomous-driving

Environment
Labtop with Nvidia Graphic card
  Ubuntu20.04
  ROS Noetic

Collecting Data
rosrun opencv_ros yolo_publisher.py
rosrun opencv_ros synchronize_save_acc.py

Training
learning.py

Experiment
rosrun opencv_ros yolo_publisher.py
rosrun opencv_ros end_to_end_driving.py
