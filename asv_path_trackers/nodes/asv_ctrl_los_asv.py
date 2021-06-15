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
        self.H = 30  # horizon, asv has 30 timestep to reach the wpt
        self.timestep_counter = 1
        self.collision_counter = 0
        self.rate = dt
        self.wp = [0., 0.]
        self.wp_counter = 0  # Current waypoint
        self.goal = [0., 0.]
        self.goal_counter = 0
        self.pos = [0., 0.]
        self.last_pos = [0., 0.]
        # self.dist_to_goal = 0.
        self.goal_dist_last = 0.
        self.new_wpt = False
        self._first_draw = True
        self._first_odom = True

        # ROS related
        self._cmd_publisher = rospy.Publisher(
            "cmd_vel", geometry_msgs.msg.Twist, queue_size=1)
        self._wps_publisher = rospy.Publisher(
            "waypoint", Marker, queue_size=10)
        self._goal_publisher = rospy.Publisher(
            "goal", Marker, queue_size=10)

        self._odom_subscriber = rospy.Subscriber(
            "state", nav_msgs.msg.Odometry, self._odom_callback, queue_size=1)
        self._obstacles_sub = rospy.Subscriber(
            "/obstacle_states", StateArray, self._obsts_callback, queue_size=1)

        self.wpt_server = actionlib.SimpleActionServer(
            "go_waypoint", GoWaypointAction, self.execute_waypoint, False)
        self.set_goal_srv = rospy.Service(
            "set_goal", SetGoal, self._set_goal_callback)

        self.obstacles = StateArray()
        self.odom = nav_msgs.msg.Odometry()
        self.cmd = geometry_msgs.msg.Twist()
        self.cmd.linear.x = 0.

        self.go_wpt_result = GoWaypointResult()
        self.wpt_server.start()

        rospy.spin()

    
    def _odom_callback(self, data):
        self.odom = data
        self.pos = [self.odom.pose.pose.position.x, 
                    self.odom.pose.pose.position.y]
        if self._first_odom:
            self.last_pos = copy.deepcopy(self.pos)
            self._first_odom = False
    
    def _obsts_callback(self, msg):
        self.obstacles = msg

    def _set_goal_callback(self, req):
        self.goal[0] = req.x
        self.goal[1] = req.y
        self.goal_dist_last = self.get_dist_to_goal()
        self._visualize_goal()
        return SetGoalResponse(True)
    
    def get_dist_to_wpt(self):
        return np.hypot(self.pos[0] - self.wp[0], self.pos[1] - self.wp[1])

    def get_dist_to_goal(self):
        return np.hypot(self.pos[0] - self.goal[0], self.pos[1] - self.goal[1]) 
    
    def _generate_cmd(self):
        # flag: 0-nothing, 1-reach goal, 2-collision, 3-far away from the goal 
        flag = 0 

        # 1. check if reach the goal
        dist_to_goal = self.get_dist_to_goal()
        if dist_to_goal < 10.:
            flag = 1
            self.goal_counter += 1
            print("ASV({}, {}) reached the {}th goal({}, {})".format(
                self.pos[0], self.pos[1], self.goal_counter, self.goal[0], self.goal[1]))
            return 0, 0, flag

        # 2. check if collide with the other ships
        if len(self.obstacles.states) > 0:  
            for ts in self.obstacles.states:
                dist_to_ts = np.hypot(self.pos[0] - ts.x, self.pos[1] - ts.y)
                if dist_to_ts < 10.+10.:
                    flag = 2
                    self.collision_counter += 1
                    print("ASV({}, {}) {}th collides with other ships({}, {})".format(
                        self.pos[0], self.pos[1], self.collision_counter, ts.x, ts.y))
                    return 0, 0, flag       

        # 3. check if go beyond the area limitation
        if abs(self.pos[0]) > 700. or abs(self.pos[1]) > 700.:
            flag = 3
            print("ASV({}, {}) go beyond the area limitation".format(
                   self.pos[0], self.pos[1]))
            return 0, 0, flag

        if self.timestep_counter > 20:
            print("This horizon is finished. ASV at (%.2f, %.2f), wpt is (%.2f, %.2f)!" % (
                self.pos[0], self.pos[1], self.wp[0], self.wp[1]))
            self.last_pos = copy.deepcopy(self.pos)
            self.wp_counter += 1
            flag = 4
            return self.u_d, self.psi_d, flag 

        if self.new_wpt:
            # get new wpt to calculate the new course
            new_course = np.arctan2(self.wp[1] - self.last_pos[1],
                                    self.wp[0] - self.last_pos[0])
            if np.abs(new_course - self.Xp) > np.pi/4.0:
                self.e_integral = 0.0
            self.Xp = new_course
            self.new_wpt = False

        xk = self.wp[0]
        yk = self.wp[1]

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

        # Publish cmd_vel
        self.cmd.linear.x = u_d
        self.cmd.angular.y = psi_d
        self.cmd.angular.z = 0.0
        self._cmd_publisher.publish(self.cmd)

        return flag

    def execute_waypoint(self, wp):
        r = rospy.Rate(1/self.rate)
        flag = 0
        done = False
        reward = 0.

        self.wp[0] = wp.x
        self.wp[1] = wp.y
        print("\nreceive and execute the waypoint: {}, {}".format(self.wp[0], self.wp[1]))
        self._visualize_waypoints()
        self.new_wpt = True

        goal_dist_now = self.get_dist_to_goal()
        reward_to_goal = 0.05 * (self.goal_dist_last - goal_dist_now)
        print("goal dist - last: %f, now: %f" % (self.goal_dist_last, goal_dist_now))
        while not rospy.is_shutdown() and flag == 0 and not done:
            # done: 0-nothing, 1-reach goal, 2-collision, 
            # 3-far away from the goal, 4-timestep is reach the horizon
            flag = self._update()
            if flag == 0:
                reward -= 0.01
                self.timestep_counter += 1
                done = False
            else:
                self.timestep_counter = 1
                if flag == 1:
                    reward += 10.
                    done = True
                elif flag == 2:
                    reward -= 20.
                    done = True
                elif flag == 3:
                    reward -= 0.02
                    done = True
                elif flag == 4:
                    reward -= 0.02
                    done = False
            r.sleep()

        self.goal_dist_last = copy.deepcopy(goal_dist_now)
        self.go_wpt_result.done = done
        self.go_wpt_result.reward = reward + reward_to_goal
        print("done: %r, reward los: %f, to_go: %f, other: %f" % (
            done, self.go_wpt_result.reward, reward_to_goal, reward))
        self.wpt_server.set_succeeded(self.go_wpt_result)

    def _visualize_waypoints(self):
        mk = Marker()
        mk.header.seq += 1
        mk.header.frame_id = "map"
        mk.header.stamp = rospy.Time.now()

        mk.ns = "waypoint"
        mk.id = 0
        mk.type = Marker.CYLINDER
        mk.scale.x = 5
        mk.scale.y = 5
        mk.scale.z = 1.  # height [m]
        mk.action = Marker.ADD

        mk.pose = geometry_msgs.msg.Pose()
        mk.pose.position.x = self.wp[0]
        mk.pose.position.y = self.wp[1]
        mk.pose.orientation.w = 1

        mk.lifetime = rospy.Duration()
        mk.color.a = .3
        mk.color.r = 0.
        mk.color.g = 1.
        mk.color.b = 0.
        self._wps_publisher.publish(mk)
    
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
        mk.color.r = 1.
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
