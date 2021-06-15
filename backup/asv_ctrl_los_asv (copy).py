#!/usr/bin/env python
"""(Integral) Line of Sight implementation for ROS

This implementation is based on [1].

TODO's:
- The LOS-controller should use the initial position as the first 'waypoint'

[1]: Handbook of Marine Craft Hydrodynamics and Motion Control. T.I. Fossen, 2011.

"""

import math as m
import numpy as np
import rospy
import copy
import geometry_msgs.msg
import nav_msgs.msg
from visualization_msgs.msg import Marker
from asv_msgs.msg import Waypoint
from std_msgs.msg import Bool
from utils import Controller

import actionlib
from asv_msgs.msg import GoWaypointAction, GoWaypointResult, GoWaypointFeedback
from asv_msgs.srv import SetGoal, SetGoalResponse


class LOSGuidanceROS(object):
    """A ROS wrapper for LOSGuidance()"""

    def __init__(self,
                 R2=20,
                 u_d=2.0,
                 de=50.0,
                 Ki=0.0,
                 dt=0.2,
                 max_integral_correction=np.pi*20.0/180.0,
                 switch_criterion='circle'):

        self.controller = LOSGuidance(R2,
                                      u_d,
                                      de,
                                      Ki,
                                      dt,
                                      max_integral_correction,
                                      switch_criterion)

        self.H = 30
        self.rate = dt
        # self.wp = self.controller.wp
        self.cwp = 0

        self._cmd_publisher = rospy.Publisher(
            "cmd_vel", geometry_msgs.msg.Twist, queue_size=1)
        self._wps_publisher = rospy.Publisher(
            "waypoints", Marker, queue_size=10)

        self._odom_subscriber = rospy.Subscriber(
            "state", nav_msgs.msg.Odometry, self._odom_callback, queue_size=1)
        self._obstacles_sub = rospy.Subscriber(
            "/obstacle_states", StateArray, self._obsts_callback, queue_size=1)

        self.wpt_server = actionlib.SimpleActionServer(
            "go_waypoint", GoWaypointAction, self.execute_waypoint, False)
        self.set_goal_srv = rospy.Service(
            "set_goal", SetGoal, self._set_goal_callback)

        self.odom = nav_msgs.msg.Odometry()
        self.cmd = geometry_msgs.msg.Twist()
        self.cmd.linear.x = u_d

        self._first_draw = True

        self.go_wpt_result = GoWaypointResult()
        self.wpt_server.start()

        rospy.spin()


    def _visualize_waypoints(self, switched):
        if not switched and not self._first_draw:
            return

        if self._first_draw:
            # for wp in range(1, self.nwp):
            mk = Marker()
            mk.header.seq += 1
            mk.header.frame_id = "map"
            mk.header.stamp = rospy.Time.now()

            mk.ns = "waypoints"
            mk.id = 0
            mk.type = Marker.CYLINDER
            D = np.sqrt(self.controller.R2)
            mk.scale.x = D
            mk.scale.y = D
            mk.scale.z = 2.  # height [m]
            mk.action = Marker.ADD

            mk.pose = geometry_msgs.msg.Pose()
            mk.pose.position.x = self.controller.wp[0]
            mk.pose.position.y = self.controller.wp[1]
            mk.pose.orientation.w = 1

            mk.lifetime = rospy.Duration()
            mk.color.a = .3
            mk.color.r = 0.
            mk.color.g = 0.
            mk.color.b = 1.

            self._wps_publisher.publish(mk)
        else:
            # for wp in [self.cwp-1, self.cwp]:
            mk = Marker()
            mk.header.seq += 1
            mk.header.frame_id = "map"
            mk.header.stamp = rospy.Time.now()

            mk.ns = "waypoints"
            mk.id = 1
            mk.type = Marker.CYLINDER
            D = np.sqrt(self.controller.R2)
            mk.scale.x = D
            mk.scale.y = D
            mk.scale.z = 2.  # height [m]
            mk.action = Marker.ADD

            mk.pose = geometry_msgs.msg.Pose()
            mk.pose.position.x = self.controller.wp[0]
            mk.pose.position.y = self.controller.wp[1]
            mk.pose.orientation.w = 1

            mk.lifetime = rospy.Duration()
            mk.color.a = .3
            mk.color.r = 0.
            mk.color.g = 0.
            mk.color.b = 1.

            self._wps_publisher.publish(mk)

        self._first_draw = True

    def _odom_callback(self, data):
        self.odom = data

    def _set_goal_callback(self, req):
        self.controller.goal[0] = req.x
        self.controller.goal[1] = req.y
        self.controller.goal[2] = req.yaw
        return SetGoalResponse(True)

    def _update(self):
        u_d, psi_d, switched = self.controller.update(
            self.odom.pose.pose.position.x, self.odom.pose.pose.position.y)

        if switched:
            print("Switched!")
            self.cwp += 1

        # Publish cmd_vel
        self.cmd.linear.x = u_d
        self.cmd.angular.y = psi_d
        self.cmd.angular.z = 0.0

        self._cmd_publisher.publish(self.cmd)

        self._visualize_waypoints(switched)
        return switched

    def execute_waypoint(self, wp):
        r = rospy.Rate(1/self.rate)
        reached = False
        counter = 0
        reward = 0.

        self.controller.wp[0] = wp.x
        self.controller.wp[1] = wp.y
        self.controller.wp_initialized = True
        self.controller.new_wpt = True

        while not rospy.is_shutdown() and not reached and counter < self.H:
            reached = self._update()
            reward += -0.01
            counter += 1
            r.sleep()

        self.go_wpt_result.done = reached
        self.go_wpt_result.reward = reward
        self.wpt_server.set_succeeded(self.go_wpt_result)


class LOSGuidance(Controller):
    """This class implements the classic LOS guidance scheme."""

    def __init__(self,
                 R2=20,
                 u_d=2.0,
                 de=50.0,
                 Ki=0.0,
                 dt=0.2,
                 max_integral_correction=np.pi*20.0/180.0,
                 switch_criterion='circle'):
        self.R2 = R2  # Radii of acceptance (squared)
        self.de = de  # Lookahead distance

        self.dt = dt
        self.max_integral_correction = np.abs(
            np.tan(max_integral_correction) * de)
        self.Ki = Ki
        self.e_integral = 0.0

        self.cWP = 0  # Current waypoint
        self.cGoal = 0
        self.wp = [0., 0.]
        self.last_x = 0.
        self.last_y = 0.
        self.wp_initialized = False
        self.new_wpt = False
        self.goal = [0., 0., 0.]

        if switch_criterion == 'circle':
            self.switching_criterion = self.circle_of_acceptance
        elif switch_criterion == 'progress':
            self.switching_criterion = self.progress_along_path

        self.Xp = 0.0
        self.u_d = u_d

    def __str__(self):
        return """Radii: %f\nLookahead distance: %f\nCurrent Waypoint: %d""" % (self.R, self.de, self.cWP)

    def circle_of_acceptance(self, x, y):
        return np.hypot(x - self.wp[0], y - self.wp[1]) < self.R2

    def check_reach_goal(self, x, y):
        return np.hypot(x - self.goal[0], y - self.goal[1]) < self.R2

    def progress_along_path(self, x, y):
        return \
            np.abs((self.wp[0] - x)*np.cos(self.Xp) +
                   (self.wp[1] - y)*np.sin(self.Xp)) < self.R2

    def update(self, x, y):
        if not self.wp_initialized:
            print("Error. No waypoints!")
            return 0, 0, False

        # if self.R2 > 999999:
        #     # Last waypoint has been reached.
        #     return 0, self.Xp, False

        # print self.wp[self.cWP,:], str(self)
        switched = False

        # bb = (x - self.wp[0])**2 + (y - self.wp[1])**2
        # print("switching criterion dist: ", bb)
        if self.switching_criterion(x, y):
            while self.switching_criterion(x, y):
                dist_to_goal = np.hypot(x - self.goal[0], y - self.goal[1])
                print("check reach goal: {}, x-{}, y-{}, goal_x-{}, goal_y-{}".format(
                    dist_to_goal, x, y, self.goal[0], self.goal[1]))

                if dist_to_goal > self.R2:
                    print("Waypoint %d: (%.2f, %.2f) reached!" % (
                        self.cWP, self.wp[0], self.wp[1]))
                    self.last_x = copy.deepcopy(self.wp[0])
                    self.last_y = copy.deepcopy(self.wp[1])

                    self.cWP += 1
                    switched = False
                else:
                    # Last waypoint reached
                    # if self.R2 < 50000:
                    print("Goal %d: (%.2f, %.2f) reached!" % (
                        self.cGoal, self.goal[0], self.goal[1]))
                    # self.R2 = np.Inf
                    self.cGoal += 1
                    return 0, self.Xp, True

        if self.new_wpt:
            # get new wpt to calculate the new course
            new_course = np.arctan2(self.wp[1] - self.last_y,
                                    self.wp[0] - self.last_x)
            if (np.abs(new_course - self.Xp) > np.pi/4.0):
                self.e_integral = 0.0
            self.Xp = new_course
            self.new_wpt = False

        xk = self.wp[0]
        yk = self.wp[1]

        # Cross-track error Eq. (10.10), [Fossen, 2011]
        e = -(x - xk)*np.sin(self.Xp) + (y - yk)*np.cos(self.Xp)
        self.e_integral += e*self.dt

        if self.e_integral*self.Ki > self.max_integral_correction:
            self.e_integral -= e*self.dt

        Xr = np.arctan2(-(e + self.Ki*self.e_integral), self.de)

        psi_d = self.Xp + Xr

        return self.u_d, psi_d, switched


if __name__ == "__main__":
    rospy.init_node("LOS_Guidance_controller")

    waypoints = rospy.get_param("~waypoints")
    u_d = rospy.get_param("~u_d", 3.0)
    R2 = rospy.get_param("~acceptance_radius", 20)
    dt = rospy.get_param("~update_rate", 0.1)
    de = rospy.get_param("~lookahead_distance", 40.0)
    Ki = rospy.get_param("~integral_gain", 0.0)
    max_integral_correction = rospy.get_param(
        "~max_integral_correction", np.pi*20/180)

    guide = LOSGuidanceROS(R2,
                           u_d,
                           de,
                           Ki,
                           dt,
                           max_integral_correction,
                           switch_criterion='circle')
