#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

if __name__ == '__main__':
    rospy.init_node("go_straight_test")
    rospy.loginfo("Node has been started.")

    pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    rate = rospy.Rate(5)

    while not rospy.is_shutdown():
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = 0.0
        pub.publish(msg)
        rate.sleep()