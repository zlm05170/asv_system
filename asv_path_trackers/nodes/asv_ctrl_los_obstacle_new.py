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
from geometry_msgs.msg import PoseStamped
import nav_msgs.msg
from visualization_msgs.msg import Marker
from asv_msgs.msg import Waypoint
from std_msgs.msg import Bool
from utils import Controller

import actionlib
from asv_msgs.msg import GoWaypointAction, GoWaypointResult, GoWaypointFeedback, StateArray
from asv_msgs.srv import SetGoal, SetGoalResponse


class LOSGuidanceROS(object):
    """A ROS wrapper for LOSGuidance()"""

    def __init__(self,
                 R2=20,
                 u_d=3.0,
                 de=50.0,
                 Ki=0.0,
                 dt=0.2,
                 max_integral_correction=np.pi*20.0/180.0
                 ):
        # params to compute the LOS
        self.R2 = R2  # Radii of acceptance (squared)
        self.u_d = u_d
        self.psi_d = 0
        self.de = de  # Lookahead distance
        self.Ki = Ki
        self.dt = dt
        self.max_integral_correction = np.abs(
            np.tan(max_integral_correction) * de)
        self.e_integral = 0.0    
        self.Xp = 0.0

        # params to train the asv
        self.H = 20  # horizon, asv has 30 timestep to reach the wpt
        self.timestep_counter = 1
        self.collision_counter = 0
        self.rate = dt
        self.goal = [0., 0.]
        self.goal_counter = 0
        self.last_pos = [0., 0.]
        # self.dist_to_goal = 0.
        self.goal_dist = 0.
        self._first_draw = True
        self._first_odom = True

        # ROS related
        self._cmd_publisher = rospy.Publisher(
            "cmd_vel", geometry_msgs.msg.Twist, queue_size=1)
        self._goal_publisher = rospy.Publisher(
            "goal", Marker, queue_size=10)

        self._odom_subscriber = rospy.Subscriber(
            "/asv/pose", PoseStamped, self._asv_pose_callback, queue_size=1)
        self._odom_subscriber = rospy.Subscriber(
            "state", nav_msgs.msg.Odometry, self._odom_callback, queue_size=1)

        self.set_goal_srv = rospy.Service(
            "set_goal", SetGoal, self._set_goal_callback)

        self.shipname = rospy.get_param("~ship_init_state", "hi")
        # self.init_state = rospy.get_param(self.shipname+"/initial_state", [])

        self.asv_pose = [0., 0.]
        self.pos = [0., 0.]
        self.odom = nav_msgs.msg.Odometry()
        self.cmd = geometry_msgs.msg.Twist()
        self.cmd.linear.x = 0.

    def _odom_callback(self, data):
        self.odom = data
        self.pos = [self.odom.pose.pose.position.x, 
                    self.odom.pose.pose.position.y]
        if self._first_odom:
            self.goal = copy.deepcopy(self.pos)
            self._first_odom = False
            # self.u_d = self.odom.twist.twist.linear.x

    def _asv_pose_callback(self, msg):
        self.asv_pose = [msg.pose.position.x, msg.pose.position.y]
    
    def _set_goal_callback(self, req):
        self.goal[0] = req.x
        self.goal[1] = req.y
        print("%s received goal: %f, %f" %(self.shipname, self.goal[0], self.goal[1]))
        rospy.sleep(0.1)
        self.Xp = np.arctan2(self.goal[1] - self.pos[1],
                             self.goal[0] - self.pos[0])
        self._visualize_goal()
        return SetGoalResponse(True)
    
    def get_dist_to_goal(self):
        return np.hypot(self.pos[0] - self.goal[0], self.pos[1] - self.goal[1]) 
    
    def _generate_cmd(self):
        # flag: 0-nothing, 1-reach goal, 2-collision, 3-far away from the goal 
        flag = 0 

        # 1. check if reach the goal
        dist_to_goal = self.get_dist_to_goal()

        if dist_to_goal < 25.:
            flag = 1
            self.goal_counter += 1
            # print("{} ({}, {}) reached the {}th goal({}, {})".format(self.shipname,
            #     self.pos[0], self.pos[1], self.goal_counter, self.goal[0], self.goal[1]))
            return 0, self.Xp, flag

        # 2. check if collide with the other ships
        dist_to_asv = np.hypot(self.pos[0] - self.asv_pose[0], self.pos[1] - self.asv_pose[1])
        if dist_to_asv < 10.+10.:
            flag = 2
            self.collision_counter += 1
            print("Target ship({}, {}) {}th collides with other ships({}, {})".format(
                self.pos[0], self.pos[1], self.collision_counter, self.asv_pose[0], self.asv_pose[1]))
            return 0, self.Xp, flag       

        xk = self.goal[0]
        yk = self.goal[1]

        # Cross-track error Eq. (10.10), [Fossen, 2011]
        e = -(self.pos[0] - xk)*np.sin(self.Xp) + (self.pos[1] - yk)*np.cos(self.Xp)
        self.e_integral += e*self.dt

        if self.e_integral*self.Ki > self.max_integral_correction:
            self.e_integral -= e*self.dt

        Xr = np.arctan2(-(e + self.Ki*self.e_integral), self.de)

        self.psi_d = self.Xp + Xr

        return self.u_d, self.psi_d, flag
        
    def _update(self):
        u_d, psi_d, flag = self._generate_cmd()
        # print("%s genereate cmd: %f, %f" % (self.shipname, u_d, psi_d))

        # Publish cmd_vel
        self.cmd.linear.x = u_d
        self.cmd.angular.y = psi_d
        self.cmd.angular.z = 0.0
        self._cmd_publisher.publish(self.cmd)

        return flag

    def run_controller(self):
        r = rospy.Rate(1./self.rate)

        while not rospy.is_shutdown():
            flag = self._update()
            r.sleep()

    def _visualize_goal(self):
        mk = Marker()
        mk.header.seq += 1
        mk.header.frame_id = "map"
        mk.header.stamp = rospy.Time.now()

        mk.ns = "goal"
        mk.id = 0
        mk.type = Marker.CYLINDER
        mk.scale.x = 10.
        mk.scale.y = 10.
        mk.scale.z = 1.  # height [m]
        mk.action = Marker.ADD

        mk.pose = geometry_msgs.msg.Pose()
        mk.pose.position.x = self.goal[0]
        mk.pose.position.y = self.goal[1]
        mk.pose.orientation.w = 1

        mk.lifetime = rospy.Duration()
        mk.color.a = .5
        mk.color.r = 0.
        mk.color.g = 0.
        mk.color.b = 0.
        self._goal_publisher.publish(mk)

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
                           max_integral_correction)

    guide.run_controller()
