# The ASV System Package

The Autonomous Surface Vehicle (ASV) System Package is a collection of ROS
packages developed by Thomas Stenersen as a part of his master's thesis.
I am using the simulator and adding a simulation based mpc controller and DRL controller.


## Contents
This package contains:
+ `asv_ctrl_vo`: an implementation of the "Velocity Obstacle" algorithm for
  collision avoidance.

+ `asv_sb_mpc` : an implementation of a simulation based model predictive control
  algorithm for collision avoidance.

+ `asv_path_trackers`: implements the (Integral) Line of Sight (LOS) method and
  a simple pure pursuit scheme for path following.

+ `asv_msgs`: message types used in the system.

+ `asv_obstacle_tracker`: package that acts as a "black box", providing
  information about the states (and possibly metadata) that a collision avoidance
  system can subscribe to. _It does not actually track obstacles._ It is also
  possible to simulate the addition of sensor noise using this package.

+ `asv_simulator`: simulates a nonlinear 3DOF surface vessel.

+ `asv_system`: metapackage with launch files and more!

+ `state_estimator`: unfinished package for estimating the ASV pose given GPS
  and IMU data.

##  Tips

<!-- * start_state: 就是设置asv_simulator里面的initial state，采用setState函数
* goal: 通过service设置，在每一轮开始时设置,，在los_asv.py中
* waypoint: 就是用actionlib发送goal，等待执行完，接收执行结果
*  -->
