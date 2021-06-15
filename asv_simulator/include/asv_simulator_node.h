#ifndef ASV_SIMULATOR_NODE_H
#define ASV_SIMULATOR_NODE_H

#include <sensor_msgs/LaserScan.h>

#include "asv_msgs/StateArray.h"
#include "asv_msgs/SetStart.h"
#include "asv_simulator.h"
#include "boost/thread.hpp"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/ros.h"
#include "tf/transform_broadcaster.h"

class VesselNode
{
public:
  VesselNode();
  void initialize(tf::TransformBroadcaster *tf, ros::Publisher *pose_pub, ros::Publisher *odom_pub,
                  ros::Publisher *path_pub, ros::Publisher *noise_pub, ros::Publisher *marker_pub,
                  ros::Publisher *circles_pub, ros::Subscriber *cmd_sub, ros::Subscriber *obstc_sub, 
                  ros::ServiceServer *set_start_srv, Vessel *vessel);
  void publishData();
  void start();
  void cmdCallback(const geometry_msgs::Twist::ConstPtr &msg);
  void obstacleCallback(const asv_msgs::StateArray::ConstPtr &msg);

  bool setStartCallback(asv_msgs::SetStart::Request &req, asv_msgs::SetStart::Response &res);

  ~VesselNode();

  std::string tf_name;

private:
  void pubMarker(const geometry_msgs::Pose &pose);

  Vessel *theVessel_;

  bool initialized_;

  tf::TransformBroadcaster *tf_;
  ros::Publisher *pose_pub_;
  ros::Publisher *odom_pub_;
  ros::Publisher *path_pub_;
  ros::Publisher *noise_pub_;
  ros::Publisher *marker_pub_;
  ros::Publisher *circles_pub_;

  ros::Subscriber *cmd_sub_;
  ros::Subscriber *obsts_sub_;

  ros::ServiceServer *set_start_srv_;

  nav_msgs::Path path_;

  double u_d_;
  double psi_d_;
  double r_d_;

  std::vector<asv_msgs::State> obstacles_;
};

#endif
