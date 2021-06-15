#!/usr/bin/env python3

import copy
import math as m
import numpy as np
import rospy
import gym
import pprint
from gym import spaces
from gym.utils import seeding
from tf import transformations

from geometry_msgs.msg import (Point, Pose, PoseStamped, Quaternion,
                               TransformStamped)
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan

from asv_msgs.msg import GoWaypointAction, GoWaypointGoal, State, StateArray
from asv_msgs.srv import SetGoal, SetGoalRequest, SetStart, SetStartRequest

import actionlib


class AsvEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(AsvEnv, self).__init__()

        # self.num_obstacle_asv = 3
        # rospy.init_node("wp_env")
        self._asv_pose_sub = rospy.Subscriber(
            "/asv/pose", PoseStamped, self._asv_pose_callback, queue_size=1)
        self._obstacles_sub = rospy.Subscriber(
            "/obstacle_states", StateArray, self._obsts_callback, queue_size=1)
        self._laser_sub = rospy.Subscriber(
            "/scan", LaserScan, self._laser_callback, queue_size=1)

        self.scan = LaserScan()
        self.start = []
        self.start_heading = 0.
        self.goal = []
        self.goal_matrix = []
        self.state_scan = []
        self.state_goal = []
        self.state_laction = []
        self.obstacles = StateArray()
        self.asv_trans = []
        self.asv_rot = []
        self.reset_id = 0

        self.heading = 0.
        self.cross_err = 0.
        self.psi_err = 0.
        self.path_dist = 0.
        self.min_dist = 0.
        self.min_angle = 0.

        self.ref_paths = []
        self.ref_path_markers = MarkerArray()
        self.ref_paths_markers_pub = rospy.Publisher(
            'ref_paths', MarkerArray, queue_size=1)

        self.wp_client = actionlib.SimpleActionClient(
            "/asv/go_waypoint", GoWaypointAction)
        self.wp_client.wait_for_server()

        rospy.wait_for_service("/asv/set_goal")
        self.set_goal_client = rospy.ServiceProxy("/asv/set_goal", SetGoal)
        rospy.wait_for_service("/asv/set_start")
        self.set_start_client = rospy.ServiceProxy("/asv/set_start", SetStart)

        # for target ships
        # rospy.wait_for_service("/obstacles/ship1/set_start")
        # self.ts1_start_client = rospy.ServiceProxy("/obstacles/ship1/set_start", SetStart)
        # rospy.wait_for_service("/obstacles/ship1/set_goal")
        # self.ts1_goal_client = rospy.ServiceProxy("/obstacles/ship1/set_goal", SetGoal)


    def _asv_pose_callback(self, msg):
        self.asv_trans = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.asv_rot = [msg.pose.orientation.x, msg.pose.orientation.y, 
                        msg.pose.orientation.z, msg.pose.orientation.w]
        self.heading = transformations.euler_from_quaternion(self.asv_rot)[2]
        # print(self.asv_trans)
        # print(self.asv_rot)

    def _obsts_callback(self, msg):
        self.obstacles = msg

    def _laser_callback(self, msg):
        self.scan = msg

    def _show_ref_paths(self):
        idx = 0
        for l in self.ref_paths:
            self._add_marker(
                Marker.LINE_STRIP, 
                [0.8, 0.8, 0.8, 0.8], 
                [.5, 0., 0.], 
                points=[p for p in l], 
                ns='path',
                idx=idx)
            idx += 1
        self.ref_paths_markers_pub.publish(self.ref_path_markers)

    def _add_marker(self,
                    markerType,
                    color,
                    scale,
                    pose=None,
                    ns=None,
                    text=None,
                    points=None,
                    idx=None):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = ns
        marker.id = idx
        marker.type = markerType
        marker.action = Marker.ADD
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]
        if pose:
            marker.pose = pose
        if text:
            marker.text = text
            marker.pose.position.z += 5.
        if points:
            marker.points = []
            for p in points:
                pt = Point()
                pt.x = p[0]
                pt.y = p[1]
                pt.z = 0.
                marker.points.append(pt)
        if ns == 'path':
            self.ref_path_markers.markers.append(marker)

    def _get_state_goal(self):
        R = transformations.quaternion_matrix(self.asv_rot)
        R[0, 3] = self.asv_trans[0]
        R[1, 3] = self.asv_trans[1]
        R[2, 3] = self.asv_trans[2]

        result = np.dot(np.linalg.inv(R), self.goal_matrix)
        rotate_goal = transformations.quaternion_from_matrix(result)
        trans_goal = transformations.translation_from_matrix(result)

        goal_dist = m.hypot(trans_goal[0], trans_goal[1])
        goal_theta = m.atan2(trans_goal[1], trans_goal[0])
        goal_yaw = transformations.euler_from_quaternion(rotate_goal)[2]
        self.state_goal = np.array([goal_dist, goal_theta])
        # self.state_goal = np.array([self.goal[0], self.goal[1]])

    def _get_state_scan(self):
        scan = self.scan.ranges
        if len(self.state_scan) == 0:
            self.state_scan.append(scan)
            self.state_scan.append(scan)
            self.state_scan.append(scan)
            self.state_scan = np.array(self.state_scan)
        else:
            self.state_scan = np.vstack(
                (self.state_scan[1:], np.array(scan)))

    def _get_state(self):
        self._get_state_goal()
        self._get_state_scan()
        return self.state_scan.copy(), np.array([self.cross_err, self.psi_err, self.heading]), \
               self.state_goal.copy(), self.state_laction.copy()

    # def _get_reward(self):
    #     if np.hypot(self.asv_trans[0] - self.state_goal[0], 
    #                 self.asv_trans[1] - self.state_goal[1]) < 15.:
    #         return 6.

    #     for ts in self.obstacles:
    #         if np.hypot(self.asv_trans[0] - ts.x, 
    #                     self.asv_trans[1] - ts.y) < 23.:
    #         return -10.
    #     return 0.

    def _apply_action_xy(self, action):
        act_m = transformations.quaternion_matrix(
            [0., 0., m.sin(0 / 2.), m.cos(0 / 2)])
        act_m[0, 3] = 10.
        act_m[1, 3] = action
        act_m[2, 3] = 0.
        
        R = transformations.quaternion_matrix(self.asv_rot)
        R[0, 3] = self.asv_trans[0]
        R[1, 3] = self.asv_trans[1]
        R[2, 3] = self.asv_trans[2]

        result = np.dot(R, act_m)
        trans = transformations.translation_from_matrix(result)
        return trans[0], trans[1]

    def _apply_action_theta(self, action):
        # action is theta
        theta = action[0]
        act_m = transformations.quaternion_matrix(
            [0., 0., m.sin(0 / 2.), m.cos(0 / 2)])
        act_m[0, 3] = 10. * np.cos(theta)
        act_m[1, 3] = 10. * np.sin(theta)
        act_m[2, 3] = 0.
        
        R = transformations.quaternion_matrix(self.asv_rot)
        R[0, 3] = self.asv_trans[0]
        R[1, 3] = self.asv_trans[1]
        R[2, 3] = self.asv_trans[2]

        result = np.dot(R, act_m)
        trans = transformations.translation_from_matrix(result)
        return trans[0], trans[1]

    def _princip(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def _update_cross_and_psi(self):
        self.psi_err = np.abs(float(self._princip(self.start_heading - self.heading)))

        # a_path = m.atan2(self.goal[1] - self.start[1], 
        #                  self.goal[0] - self.start[0])
        angle = m.atan2(self.asv_trans[1] - self.start[1], 
                        self.asv_trans[0] - self.start[0])
        dist = m.hypot(self.asv_trans[0] - self.start[0], 
                       self.asv_trans[1] - self.start[1])
        self.cross_err = np.abs(m.sin(self.start_heading - angle) * dist)

    def _update_ts_in_asv(self):
        # get angle and dist of target ship in own ship's frame
        dists = []
        angles = []
        if len(self.obstacles.states) > 0:
            for ts in self.obstacles.states:                                        
                ts_matrix = transformations.euler_matrix(0., 0., ts.psi)
                ts_matrix[0, 3] = ts.x
                ts_matrix[1, 3] = ts.y
                ts_matrix[2, 3] = 0

                R = transformations.quaternion_matrix(self.asv_rot)
                R[0, 3] = self.asv_trans[0]
                R[1, 3] = self.asv_trans[1]
                R[2, 3] = self.asv_trans[2]

                result = np.dot(np.linalg.inv(R), ts_matrix)
                trans_ts = transformations.translation_from_matrix(result)
                ts_dist = m.hypot(trans_ts[0], trans_ts[1])
                ts_angle = m.atan2(trans_ts[1], trans_ts[0])
                dists.append(ts_dist)
                angles.append(ts_angle)

        self.min_dist = min(dists)
        min_id = dists.index(self.min_dist)
        self.min_angle = angles[min_id]

    def _update_colreg_reward(self):
        rew = 0.

        if self.min_angle < 0. and self.min_angle > -m.radians(112.5): # starboard
            rew = 10*np.exp(-0.07 * self.min_dist)
        else:  # port
            rew = 10*np.exp(-0.09 * self.min_dist)
        return -rew

    def _update_path_reward(self):
        reward_err_psi = 0.
        reward_err_cross = 0.
        if self.min_dist < 50.:
            if np.abs(self.psi_err) < 0.3:
                reward_err_psi = -0.3
        elif 50. < self.min_dist < 80.:
            if np.abs(self.psi_err) < 0.2:
                reward_err_psi = -0.15
        else:
            if np.abs(self.psi_err) < 0.1:
                reward_err_psi = 0.2
            elif 0.1 < np.abs(self.psi_err) < 0.3:
                reward_err_psi = 0.            
            else:
                reward_err_psi = -0.06 * np.abs(self.psi_err)
        # print("***psi error reward: ", reward_err_psi)

        if self.min_dist < 50.:
            if np.abs(self.cross_err) < 10.:
                reward_err_cross = -0.3
        elif 50. < self.min_dist < 80.:
            if np.abs(self.cross_err) < 5.:
                reward_err_cross = -0.15
        else:
            if np.abs(self.cross_err) < 1.:
                reward_err_cross = 0.2  #.5 * (1-np.abs(self.cross))
            else:
                reward_err_cross = -0.005 * np.abs(self.cross_err)

        return reward_err_psi, reward_err_cross
        # return 0.1 * ((1+np.cos(self.psi_err)) * (1+np.exp(-0.05*self.cross_err)) - 1)

    def step(self, action):
        # * description: generate a waypoint, publish it, then wait for the reach signal
        #   from the LOS
        # * input: a waypoint
        # * output: new state, reward, done or not, []
        # - how to compute the reward? let us only consider one asv firstly
        # - use service to publish waypoint and receive feedback?

        # 1. execute the action
        wpt_goal = GoWaypointGoal()
        if action == 0:
            ac = -10.
        elif action == 1:
            ac = 0.
        elif action == 2:
            ac = 10.

        wpt_goal.x, wpt_goal.y = self._apply_action_xy(ac)
        print("action: %f; pose: %f, %f; wpt: %f, %f" % (
            ac, self.asv_trans[0], self.asv_trans[1], wpt_goal.x, wpt_goal.y))
        self.wp_client.send_goal(wpt_goal)
        self.wp_client.wait_for_result()

        # 2. compute reward
        res = self.wp_client.get_result()

        # update cross and psi
        self._update_cross_and_psi()
        self._update_ts_in_asv()

        print("min dist: %f, min angle: %f" % (self.min_dist, self.min_angle))
        reward_psi, reward_cross = self._update_path_reward()
        reward_colreg = self._update_colreg_reward()
        reward = res.reward + reward_colreg + reward_psi + reward_cross
        print("reward: %f; los: %f; psi: %f, cross: %f; colreg: %f **** cross_err: %f; psi_err: %f" % (
            reward, res.reward, reward_psi, reward_cross, reward_colreg, self.cross_err, self.psi_err))

        done = res.done
        # if self.cross_err > 250. or (self._dist_to_goal() - self.path_dist) > 50.:
        #     done = True

        # 3. get the new obs, done, and info
        self.state_laction = np.array([action])
        scan, pose, goal, lac = self._get_state()

        return scan, pose, goal, lac, reward, done, {}

    def _generate_new_point(self, p, r, rad):
        return [p[0] + r*m.cos(rad),
                p[1] + r*m.sin(rad)]  

    def _dist_to_goal(self):
        return m.hypot(self.asv_trans[0] - self.goal[0], 
                       self.asv_trans[1] - self.goal[1])

    def reset(self):
        self.ref_paths= []
        # generate start and goal pose for asv
        # self.start = np.random.uniform(-100., 100., 2)
        # self.start_heading = np.random.uniform(-np.pi, np.pi, 1)
        self.start = [-50., 0.]
        self.start_heading = -1.57
        ss = State()
        ss.x = self.start[0]
        ss.y = self.start[1]
        ss.psi = self.start_heading
        ss.u = 5.
        ss.v = 0.
        ss.r = 0.
        # send start
        rospy.loginfo("[reset] start: %f, %f, %f" % (self.start[0], self.start[1], self.start_heading))
        res = self.set_start_client(ss)

        # goal_r = np.random.uniform(250., 350., 1)
        # self.path_dist = goal_r
        # self.goal = self._generate_new_point(self.start, goal_r, self.start_heading)
        self.goal = [-50., -400.]

        # send goal
        res = self.set_goal_client(self.goal[0], self.goal[1], self.start_heading)
        if res.done:
            self.goal_matrix = transformations.quaternion_matrix(
                [0., 0., m.sin(self.start_heading / 2.), m.cos(self.start_heading / 2)])
            self.goal_matrix[0, 3] = self.goal[0]
            self.goal_matrix[1, 3] = self.goal[1]
            self.goal_matrix[2, 3] = 0.
        rospy.loginfo("[reset] goal: %f, %f, %f" % (self.goal[0], self.goal[1], self.start_heading))
        self.ref_paths.append([[ss.x, ss.y], [self.goal[0], self.goal[1]]])

        # a_path = m.atan2(self.goal[1] - self.start[1], 
        #                  self.goal[0] - self.start[0])
        # rospy.loginfo("a_path: %f; start_heading: %f" % (a_path, self.start_heading))

        self.reset_id = 5
        # self.reset_id = 5
        # generate starts and goals for target ships
        # crossing from right
        if self.reset_id == 0:
            ts_l = np.random.uniform(80, 150, 1)[0]
            cp = self._generate_new_point(self.start, ts_l, self.start_heading)
            ts_h = np.random.uniform(self.start_heading-m.radians(20.), self.start_heading-m.radians(112.5), 1)[0]
            ts_s = self._generate_new_point(cp, ts_l, ts_h)
            ts_g = self._generate_new_point(ts_s, ts_l*2, ts_h-m.radians(180.))
            ts_yaw = m.atan2(ts_g[1]-ts_s[1], ts_g[0]-ts_s[0])        

            # send starts and goals for target ships
            ss.x = ts_s[0]
            ss.y = ts_s[1]
            ss.psi = ts_yaw
            res = self.ts1_start_client(ss)
            # send ts goal           
            res = self.ts1_goal_client(ts_g[0], ts_g[1], ts_yaw)
            # ss.x = 0
            # ss.y  = -50
            # ss.psi = 1
            # res = self.ts1_start_client(ss)
            # ts_g = [100, 100]
            # ts_yaw= ss.psi
            # res = self.ts1_goal_client(ts_g[0], ts_g[1], ts_yaw)

        # head on
        elif self.reset_id == 1:
            ss.x = self.goal[0]
            ss.y = self.goal[1]
            ss.psi = self.start_heading - m.radians(180.)
            res = self.ts1_start_client(ss)
            ts_g = [self.start[0], self.start[1]]
            res = self.ts1_goal_client(ts_g[0], ts_g[1], ss.psi)

        # overtaking
        elif self.reset_id == 2:
            d = np.random.uniform(60, 100, 1)[0]
            cp = self._generate_new_point(self.start, d, self.start_heading)
            ss.x = cp[0]
            ss.y = cp[1]
            ss.psi = self.start_heading
            ss.u = 1.
            res = self.ts1_start_client(ss)

            ts_g = self._generate_new_point(self.start, d+60, ss.psi)
            res = self.ts1_goal_client(ts_g[0], ts_g[1], ss.psi)

        # crossing from left
        elif self.reset_id == 3:
            ts_l = np.random.uniform(60, 120, 1)[0]
            cp = self._generate_new_point(self.start, ts_l, self.start_heading)
            ts_h = np.random.uniform(self.start_heading+m.radians(50.), self.start_heading+m.radians(112.5), 1)[0]
            ts_s = self._generate_new_point(cp, ts_l, ts_h)
            ts_g = self._generate_new_point(ts_s, ts_l*2, ts_h-m.radians(180.))
            ts_yaw = m.atan2(ts_g[1]-ts_s[1], ts_g[0]-ts_s[0])        

            # send starts and goals for target ships
            ss.x = ts_s[0]
            ss.y = ts_s[1]
            ss.psi = ts_yaw
            res = self.ts1_start_client(ss)
            # send ts goal           
            res = self.ts1_goal_client(ts_g[0], ts_g[1], ts_yaw)

                       
        # self.ref_paths.append([[ss.x, ss.y], [ts_g[0], ts_g[1]]])
        self._show_ref_paths()
        self.reset_id += 1

        self._update_cross_and_psi()
        self._update_ts_in_asv()

        self.state_laction = np.array([0.])
        scan, pose, goal, lac = self._get_state()
        return scan, pose, goal, lac
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# if __name__ == "__main__":
#     rospy.init_node("wp_env")
#     env = AsvEnv()
#     r = rospy.Rate(1)

#     while not rospy.is_shutdown():
#         obs = env.reset()
#         print("hi")
#         # act = agent.act(obs)
#         action = [50., 0.]
#         env.step(action)
#         # r.sleep()
