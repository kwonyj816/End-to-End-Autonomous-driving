🖥 Environment

Hardware

Laptop with NVIDIA GPU

Software

Ubuntu 20.04

ROS Noetic

📊 Data Collection

Run the following nodes to detect objects and synchronize sensor data:

rosrun opencv_ros yolo_publisher.py
rosrun opencv_ros synchronize_save_acc.py


Description

yolo_publisher.py : Real-time object detection using YOLO

synchronize_save_acc.py : Time-synchronized data logging for training

🧠 Model Training

Train the end-to-end driving network:

python learning.py

🚀 Experiment / Inference

Run the perception and autonomous control pipeline:

rosrun opencv_ros yolo_publisher.py
rosrun opencv_ros end_to_end_driving.py


Description

end_to_end_driving.py : Predicts driving commands directly from perception outputs

📌 Project Workflow
Sensor Data → YOLO Detection → Synchronized Dataset → Deep Learning Training → End-to-End Driving Control
