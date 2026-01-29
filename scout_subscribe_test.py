#!/usr/bin/env python3
import rospy
from scout_msgs.msg import ScoutStatus
from scout_msgs.msg import ScoutBmsStatus
def vel_callback(msg: ScoutStatus):
    rospy.loginfo("linear_velocity: "+str(msg.linear_velocity)
                  +", angular_velocity: "+str(msg.angular_velocity))
    rospy.loginfo("light_mode: "+str(msg.front_light_state.mode))
if __name__ == '__main__':
    rospy.init_node("subscribe_test")

    sub = rospy.Subscriber("/scout_status", ScoutStatus, callback=vel_callback)

    rospy.loginfo("Node has been started.")
    rospy.spin()