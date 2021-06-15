#ifndef ASV_CTRL_SB_MPC
#define ASV_CTRL_SB_MPC

#include <Eigen/Dense>

#include "asv_msgs/State.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/Odometry.h"
#include "visualization_msgs/Marker.h"
#include "asv_ctrl_sb_mpc/obstacle.h"
#include "asv_ctrl_sb_mpc/shipModel.h"

class simulationBasedMpc
{

	
	public:
		/// Constructor
		simulationBasedMpc();
		/// Destructor
		~simulationBasedMpc();
		
		/**
		* @brief Initializes the controller.
		*
		* @param obstacles A pointer to a vector of obstacles provided by the ROS 
		* wrapper (by subscribing to an obstacle tracker node).
		* @param map A pointer to the occupancy grid published by the map_server.
		*/
		void initialize(std::vector<asv_msgs::State> *obstacles, nav_msgs::OccupancyGrid *map);
		
		/**
		* @brief Callback for updating the internal ASV state (data provided by ROS wrapper).
		*
		* @param msg The Odometry message which contains the state data.
		* @param u_d The desired surge speed set point provided by, e.g., a LOS algorithm.
		* @param psi_d The desired heading set point provided by, e.g., a LOS algorithm.
		*/		
		void updateAsvState(const nav_msgs::Odometry::ConstPtr &msg,
							const double &u_d,
							const double &psi_d);
		
		/**
		* @brief Called after the velocity field has been updated to get the (u, psi) pair
		* with the lowest cost.
		*
		* @param u_best The reference parameter to store the "best" surge speed.
		* @param psi_best The reference parameter to store the "best" heading.
		*/
		void getBestControlOffset(double &u_d_best, double &psi_d_best);
		
		
		private:
				
		Eigen::Vector3d asv_pose_;
		Eigen::Vector3d asv_twist_;
		double u_d_;
		double psi_d_;
				
		std::vector<asv_msgs::State> *obstacles_;
		nav_msgs::OccupancyGrid *map_;
		nav_msgs::OccupancyGrid local_map_;
		ros::Publisher lm_pub;

		double Chi_ca_last_;
		double P_ca_last_;
		
		std::vector<double> Chi_ca_;
		std::vector<double> P_ca_;
		
		std::vector<obstacle*> obstacles_vect;

		shipModel *asv;
		double costFnc(double P_ca, double Chi_ca, int k);
		double Delta_P(double P_ca);
		double Delta_Chi(double Chi_ca);
		double cost;
			
		// Cost function weights;
		const double P_;
		const double Q_;
		const double D_CLOSE_;
		const double D_SAFE_;
		const double K_COLL_;
		const double PHI_AH_;
		const double PHI_OT_;
		const double PHI_HO_;
		const double PHI_CR_;
		const double KAPPA_;
		const double K_P_;
		const double K_CHI_;
		const double K_DP_;
		const double K_DCHI_SB_;
		const double K_DCHI_P_;
		
		const double T_; 	// Prediction horizon
		const double DT_;	// Step length, obstacle trajectory prediction 		
		
		int n_samp;

			
};

#endif 
