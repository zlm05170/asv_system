#include <ros/console.h>

#include <cmath>

#include "asv_ctrl_sb_mpc/obstacle.h"

	

obstacle::obstacle(double x_0, double y_0, double u_0, double v_0,  double psi_0, double radius, double T, double dt)
{
	T_ = T;
	dt_ = dt;
	n_samp_ = T_/dt_;
	
	x_.resize(n_samp_);
	y_.resize(n_samp_);
	u_.resize(n_samp_);
	v_.resize(n_samp_);

	A_ = 5;
	B_ = 5;
	C_ = 2;
	D_ = 2;
	L_ = A_ + B_;
	W_ = C_ + D_;

	psi_ = psi_0;
	radius_ = radius;

	calculatePositionOffsets();

	psi_ = psi_0;
	x_(0) = x_0 + os_x*cos(psi_) - os_y*sin(psi_);
	y_(0) = y_0 + os_x*sin(psi_) + os_y*cos(psi_);
	u_(0) = u_0;
	v_(0) = v_0;
	

	r11_ = cos(psi_);
	r12_ = -sin(psi_);
	r21_ = sin(psi_);
	r22_ = cos(psi_);

	calculateTrajectory();
};

obstacle::~obstacle()
{
};

double obstacle::getL(){
	return L_;
}

double obstacle::getW(){
	return W_;
}

void obstacle::calculatePositionOffsets(){
	os_x = A_ - B_;
	os_y = D_ - C_;
}

void obstacle::calculateTrajectory()
{
	for (int i = 1; i < n_samp_-1;  i++)
	{
		x_(i) = (x_(i-1) + (r11_*u_(i-1) + r12_*v_(i-1))*dt_);
		y_(i) = (y_(i-1) + (r21_*u_(i-1) + r22_*v_(i-1))*dt_);
		u_(i) = (u_(i-1));
		v_(i) = (v_(i-1));
	}

};



	

	
