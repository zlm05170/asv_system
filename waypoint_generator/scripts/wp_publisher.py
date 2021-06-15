#!/usr/bin/env python

import numpy as np
import rospy
from asv_msgs.msg import Waypoint
from std_msgs.msg import Bool


class WptPub(object):
    def __init__(self):
        self.idx = 0
        self.wpts = [[50., 0], [130, -30], [
            200, -50], [280, -30], [350, 0], [400, 0]]
        self.wpt_pub = rospy.Publisher(
            "/asv/wpt_target", Waypoint, queue_size=10)
        self.wpt_fd_sub = rospy.Subscriber(
            "/asv/wpt_feedback", Bool, self.wptFeedbackCall,  queue_size=1)
        self.pub()
        rospy.spin()

    def wptFeedbackCall(self, msg):
        if msg.data and self.idx < len(self.wpts):
            self.pub()

    def pub(self):
        wpt = Waypoint()
        wpt.x = self.wpts[self.idx][0]
        wpt.y = self.wpts[self.idx][1]
        i = 0
        while i < 2:
            self.wpt_pub.publish(wpt)
            print "publish waypoint"
            rospy.sleep(1)
            i += 1
        self.idx += 1


if __name__ == "__main__":
    rospy.init_node("waypoint_publisher")
    wp = WptPub()
