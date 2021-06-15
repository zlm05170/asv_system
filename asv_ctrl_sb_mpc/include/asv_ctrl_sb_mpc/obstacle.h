#ifndef OBSTACLE
#define OBSTACLE

#include  <vector>
#include <Eigen/Dense>

class obstacle
{
	public:
	
	/// Constructor
	obstacle(double x_0, double y_0, double u_0, double v_0,  double psi_0, double radius, double T, double dt);
	
	/// Destructor
	~obstacle();

	Eigen::VectorXd x_;
	Eigen::VectorXd y_;
	Eigen::VectorXd u_;
	Eigen::VectorXd v_;

	double psi_;
	double radius_;
	double A_, B_, D_, C_, L_, W_;
	double os_x, os_y;

	double getL();
	double getW();
	
	private:
	
	void calculatePositionOffsets();
	void calculateTrajectory();
	
	int n_samp_;
	double T_;
	double dt_;

	//Elements of the rotation matrix
	double r11_, r12_, r21_, r22_;

};

#endif
