#include "ros/ros.h"
#include <ros/console.h>

#include <vector>
#include <ctime>

#include "nav_msgs/OccupancyGrid.h"
#include "asv_msgs/StateArray.h"
#include "asv_msgs/Offset.h"

#include "asv_ctrl_sb_mpc/asv_ctrl_sb_mpc.h"
#include "asv_ctrl_sb_mpc/asv_ctrl_sb_mpc_node.h"

int main(int argc, char *argv[])
{
	ros::init(argc, argv, "asv_ctrl_sb_mpc_node");
	ros::start();
	
	ROS_INFO("Starting sm_mpc_node");
	
	ros::NodeHandle n;
	
	simulationBasedMpcNode sb_mpc_node;
	simulationBasedMpc *sb_mpc = new simulationBasedMpc;
	
	ros::Publisher cmd_pub = n.advertise<geometry_msgs::Twist>("asv/cmd_vel",10);
	
	ros::Publisher os_pub = n.advertise<asv_msgs::Offset>("asv/offset",10);

	ros::Subscriber obstacle_sub = n.subscribe("obstacle_states",
											   1,
											   &simulationBasedMpcNode::obstacleCallback,
											   &sb_mpc_node);
											   
	ros::Subscriber og_sub = n.subscribe("/map",
										 1,
										 &simulationBasedMpcNode::mapCallback,
										 &sb_mpc_node);
    
    ros::Subscriber asv_sub	= n.subscribe("asv/state",
										  1,
										  &simulationBasedMpcNode::asvCallback,
										  &sb_mpc_node);
	
	ros::Subscriber cmd_sub = n.subscribe("asv/LOS/cmd_vel",
										 1,
										 &simulationBasedMpcNode::cmdCallback,
										 &sb_mpc_node);
	
	sb_mpc_node.initialize(&cmd_pub, & os_pub, &obstacle_sub, &og_sub, &asv_sub, &cmd_sub, sb_mpc);
	sb_mpc_node.start();
	
	ros::shutdown();
	return 0;	
	}
	
simulationBasedMpcNode::simulationBasedMpcNode() : sb_mpc_(NULL),
												   cmd_pub_(NULL),
												   os_pub_(NULL),
												   obstacle_sub_(NULL),
												   og_sub_(NULL),
												   asv_sub_(NULL),
												   cmd_sub_(NULL) {};

simulationBasedMpcNode::~simulationBasedMpcNode() {};

void simulationBasedMpcNode::initialize(ros::Publisher *cmd_pub,
										ros::Publisher *os_pub,
										ros::Subscriber *obstacle_sub,
										ros::Subscriber *og_sub,
										ros::Subscriber *asv_sub,
										ros::Subscriber *cmd_sub,
										simulationBasedMpc *sb_mpc)
{
	cmd_pub_ = cmd_pub;
	os_pub_ = os_pub;
	obstacle_sub_ = obstacle_sub;
	og_sub_ = og_sub;
	asv_sub_ = asv_sub;
	cmd_sub_ = cmd_sub;
	
	sb_mpc_ = sb_mpc;
	u_os_ = 1;
	psi_os_ = 0;
	
	sb_mpc_->initialize(&obstacles_, &map_);
}										

void simulationBasedMpcNode::start()
{
	double rate = 10.0;
	ros::Rate loop_rate(rate);
	
	clock_t tick, tock;
	double t = 0;
	while (ros::ok())
	{
		// Run MPC every 5 seconds
		if (t > 5){
			tick = clock();
			sb_mpc_->getBestControlOffset(u_os_, psi_os_);
			tock = clock() - tick;
			t = 0;
			os_.P_ca = u_os_;
			os_.Chi_ca = psi_os_;
			os_pub_->publish(os_);
			//ROS_INFO("Runtime: %0.2f", ((float)tock)/CLOCKS_PER_SEC);
		}
		t += 1/rate;

		cmd_vel_.linear.x = u_d_*u_os_;
		cmd_vel_.angular.y = psi_d_ + psi_os_;
		cmd_pub_->publish(cmd_vel_);
		
		ros::spinOnce();
		loop_rate.sleep();
	}
}

void simulationBasedMpcNode::asvCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
	sb_mpc_->updateAsvState(msg, u_d_, psi_d_);
}

void simulationBasedMpcNode::obstacleCallback(const asv_msgs::StateArray::ConstPtr & msg)
{
	obstacles_ = msg->states;
}

void simulationBasedMpcNode::cmdCallback(const geometry_msgs::Twist::ConstPtr &msg)
{
	u_d_ = msg->linear.x;
	psi_d_ = msg->angular.y;
}

void::simulationBasedMpcNode::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr &msg)
{
  // Copy what we need
  map_.info.resolution = msg->info.resolution;
  map_.info.height = msg->info.height;
  map_.info.width = msg->info.width;
  map_.info.origin.position.x = msg->info.origin.position.x;
  map_.info.origin.position.y = msg->info.origin.position.y;

  ROS_INFO("r %f, h %d, w%d, px %f, py %f",
           map_.info.resolution,
           map_.info.height,
           map_.info.width,
           map_.info.origin.position.x,
           map_.info.origin.position.y);

  map_.data = msg->data;
}
