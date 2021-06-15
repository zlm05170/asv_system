#include <Eigen/Dense>

#include "ros/ros.h"

// Message types
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "asv_simulator.h"
#include "asv_simulator_node.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Twist.h"
#include "light_scan_sim/Circle.h"
#include "light_scan_sim/CircleList.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "tf/transform_broadcaster.h"

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "asv_simulator_node");
  ros::start();

  ROS_INFO("Started ASV Simulator node");

  ros::NodeHandle nh;
  ros::NodeHandle priv_nh("~");

  std::string name = ros::names::clean(ros::this_node::getNamespace());
  if (name.empty())
    name = "asv";

  Vessel my_vessel;

  my_vessel.initialize(priv_nh);

  VesselNode my_vessel_node;
  my_vessel_node.tf_name = name;

  tf::TransformBroadcaster tf = tf::TransformBroadcaster();
  ros::Publisher pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 10);
  ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("state", 10);
  ros::Publisher path_pub = nh.advertise<nav_msgs::Path>("path", 10);
  ros::Publisher noise_pub = nh.advertise<geometry_msgs::Vector3>("wave_noise", 10);
  ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("vessel_marker", 10);
  ros::Publisher circles_pub = nh.advertise<light_scan_sim::CircleList>("map_circles", 10);

  ros::Subscriber cmd_sub = nh.subscribe("cmd_vel", 1, &VesselNode::cmdCallback, &my_vessel_node);
  ros::Subscriber obsts_sub = nh.subscribe("obstacle_states", 1, &VesselNode::obstacleCallback, &my_vessel_node);

  ros::ServiceServer set_start_srv = nh.advertiseService("set_start", &VesselNode::setStartCallback, &my_vessel_node);

  my_vessel_node.initialize(&tf, &pose_pub, &odom_pub, &path_pub, &noise_pub, &marker_pub, &circles_pub, &cmd_sub,
                            &obsts_sub, &set_start_srv, &my_vessel);

  my_vessel_node.start();

  ros::shutdown();
  return 0;
}

VesselNode::VesselNode()
  : theVessel_(NULL)
  , initialized_(false)
  , tf_(NULL)
  , pose_pub_(NULL)
  , odom_pub_(NULL)
  , path_pub_(NULL)
  , noise_pub_(NULL)
  , marker_pub_(NULL)
  , circles_pub_(NULL)
  , cmd_sub_(NULL)
  , obsts_sub_(NULL)
  , u_d_(0.0)
  , psi_d_(0.0)
  , r_d_(0.0)
{
}

VesselNode::~VesselNode()
{
}

void VesselNode::initialize(tf::TransformBroadcaster *tf, ros::Publisher *pose_pub, ros::Publisher *odom_pub,
                            ros::Publisher *path_pub, ros::Publisher *noise_pub, ros::Publisher *marker_pub,
                            ros::Publisher *circles_pub, ros::Subscriber *cmd_sub, ros::Subscriber *obsts_sub,
                            ros::ServiceServer *set_start_srv, Vessel *vessel)
{
  if (!initialized_)
  {
    tf_ = tf;
    pose_pub_ = pose_pub;
    odom_pub_ = odom_pub;
    path_pub_ = path_pub;
    noise_pub_ = noise_pub;
    marker_pub_ = marker_pub;
    circles_pub_ = circles_pub;

    cmd_sub_ = cmd_sub;
    obsts_sub_ = obsts_sub;

    set_start_srv_ = set_start_srv;

    theVessel_ = vessel;
    initialized_ = true;
  }
  else
  {
    ROS_ERROR("Attempted to initialize VesselNode twice. Doing nothing...");
  }
}

void VesselNode::start()
{
  ros::Rate loop_rate(1.0 / theVessel_->getDT());

  while (ros::ok())
  {
    theVessel_->updateSystem(u_d_, psi_d_, r_d_);

    this->publishData();

    ros::spinOnce();
    loop_rate.sleep();
  }
}

void VesselNode::publishData()
{
  static int counter = 0;

  Eigen::Vector3d eta, nu, wave_noise;

  /// @todo This could be done with less overhead
  theVessel_->getState(eta, nu);

  tf::Transform transform;
  nav_msgs::Odometry odom;
  geometry_msgs::PoseStamped pose;

  transform.setOrigin(tf::Vector3(eta[0], eta[1], 0));

  tf::Quaternion q;
  q.setRPY(0, 0, eta[2]);
  transform.setRotation(q);

  tf_->sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", tf_name));

  odom.header.seq = counter;
  odom.header.stamp = ros::Time::now();
  odom.header.frame_id = "map";
  odom.child_frame_id = tf_name;

  odom.pose.pose.position.x = eta[0];
  odom.pose.pose.position.y = eta[1];

  tf::quaternionTFToMsg(q, odom.pose.pose.orientation);

  odom.twist.twist.linear.x = nu[0];
  odom.twist.twist.linear.y = nu[1];
  odom.twist.twist.angular.z = nu[2];
  odom_pub_->publish(odom);

  pose.header = odom.header;
  pose.pose.position = odom.pose.pose.position;
  pose.pose.orientation = odom.pose.pose.orientation;
  pose_pub_->publish(pose);

  if (path_pub_->getNumSubscribers() > 0) 
  {
    path_.header = pose.header;
    path_.poses.push_back(pose);
    path_pub_->publish(path_);
  }

  // publish marker
  pubMarker(pose.pose);

  ++counter;

  static geometry_msgs::Vector3 v3_noise;

  theVessel_->getWaveNoise(wave_noise);
  v3_noise.x = wave_noise[0];
  v3_noise.y = wave_noise[1];
  v3_noise.z = wave_noise[2];
  noise_pub_->publish(v3_noise);
}

void VesselNode::cmdCallback(const geometry_msgs::Twist::ConstPtr &msg)
{
  ROS_INFO_ONCE("Received control input!");

  u_d_ = msg->linear.x;
  psi_d_ = msg->angular.y;
  r_d_ = msg->angular.z;
}

void VesselNode::obstacleCallback(const asv_msgs::StateArray::ConstPtr &msg)
{
  // obstacles_.clear();
  obstacles_ = msg->states;
}

bool VesselNode::setStartCallback(asv_msgs::SetStart::Request& req, asv_msgs::SetStart::Response& res)
{
  path_.poses.clear();
  theVessel_->setState(req.start_state);
  res.done = true;
  return true;
}

void VesselNode::pubMarker(const geometry_msgs::Pose &pose)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.ns = tf_name;
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = 1;
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.color.r = 255.0 / 255.;
  marker.color.g = 31. / 255.;
  marker.color.b = 0.0 / 255.0;
  marker.color.a = 0.3;
  marker.scale.x = theVessel_->getR() * 2.0;
  marker.scale.y = theVessel_->getR() * 2.0;
  marker.scale.z = 0.01;

  marker.pose = pose;
  marker_pub_->publish(marker);
}