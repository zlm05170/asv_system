#include <ros/console.h>

#include "asv_ctrl_sb_mpc/shipModel.h"
#include <Eigen/Dense>

//Debug
#include <typeinfo>
#include <cmath>

const float PI = 3.1415927;
static const double DEG2RAD = PI/180.0f;
static const double RAD2DEG = 180.0f/PI;

/// Assures that the numerical difference is at most PI
double normalize_angle_diff(double angle, double angle_ref);

/// Assures that angle is between [-PI, PI)
double normalize_angle(double angle);

shipModel::shipModel(double T, double dt)
{	
	T_ = T;
	DT_ = dt;
	n_samp = floor(T_/DT_);

	x.resize(n_samp);
	y.resize(n_samp);
	psi.resize(n_samp);
	u.resize(n_samp);
	v.resize(n_samp);
	r.resize(n_samp);

	tau = Eigen::Vector3d::Zero();

	A_ = 5;
	B_ = 5;
	C_ = 2;
	D_ = 2;
	L_ = A_ + B_;
	W_ = C_ + D_;

	calculate_position_offsets();

	radius = 10.0; // [m] (In reality 4.15 m)
	M = 3980.0; // [kg]
	I_z = 19703.0; // [kg/m2]

	// Added M terms
	X_udot = 0.0;
	Y_vdot = 0.0;
	Y_rdot = 0.0;
	N_vdot = 0.0;
	N_rdot = 0.0;

	// Linear damping terms [X_u, Y_v, Y_r, N_v, N_r]
	X_u	= -50.0;
	Y_v = -200.0;
	Y_r = 0.0;
	N_v = 0.0;
	N_r = -1281;//-3224.0;

	// Nonlinear damping terms [X_|u|u, Y_|v|v, N_|r|r, X_uuu, Y_vvv, N_rrr]
	X_uu = -135.0;
	Y_vv = -2000.0;
	N_rr = 0.0;
	X_uuu = 0.0;
	Y_vvv = 0.0;
	N_rrr = -3224.0;

	Eigen::Matrix3d Mtot;
	Mtot << M - X_udot, 0, 0,
	        0, M-Y_vdot, -Y_rdot,
	        0, -Y_rdot, I_z-N_rdot;
	Minv = Mtot.inverse();

	//Force limits
	Fx_min = -6550.0;
	Fx_max = 13100.0;
	Fy_min = -645.0;
	Fy_max = 645.0;

	// Other
	rudder_d = 4.0; // distance from rudder to CG
	Kp_u = 1.0;
	Kp_psi = 5.0;
	Kd_psi = 1.0;
	Kp_r = 8.0;

}

shipModel::~shipModel(){
}

double shipModel::getL(){
	return L_;
}

double shipModel::getW(){
	return W_;
}

void shipModel::linearPrediction(Eigen::Vector3d asv_pose, Eigen::Vector3d asv_twist, double u_d, double psi_d){

	psi(0) = normalize_angle(asv_pose(2));
	x(0) = asv_pose(0) + os_x*cos(asv_pose(2)) - os_y*sin(asv_pose(2));
	y(0) = asv_pose(1) + os_x*sin(asv_pose(2)) + os_y*cos(asv_pose(2));
	u(0) = asv_twist(0);
	v(0) = asv_twist(1);
	r(0) = asv_twist(2);

	double r11, r12, r21, r22;

	r11 = cos(psi_d);
	r12 = -sin(psi_d);
	r21 = sin(psi_d);
	r22 = cos(psi_d);

	for (int i = 0; i < n_samp-1; i++){

		x(i+1) = x(i) + DT_*(r11*u(i) + r12*v(i));
		y(i+1) = y(i) + DT_*(r21*u(i) + r22*v(i));
		psi(i+1) = psi_d;
		u(i+1) = u_d;
		v(i+1) = 0;

	}
}


void shipModel::eulersMethod(Eigen::Vector3d asv_pose, Eigen::Vector3d asv_twist, double u_d, double psi_d)
{

	psi(0) = normalize_angle(asv_pose(2));
	x(0) = asv_pose(0) + os_x*cos(psi(0)) - os_y*sin(psi(0));
	y(0) = asv_pose(1) + os_x*sin(psi(0)) + os_y*cos(psi(0));
	u(0) = asv_twist(0);
	v(0) = asv_twist(1);
	r(0) = asv_twist(2);

	double x_, y_, psi_, u_, v_, r_;
	double t = 0;
	Eigen::Vector3d temp;
	double r11, r12, r21, r22; // rotation matrix elements

	for (int i = 0; i < n_samp-1; i++){

		t += DT_;

		psi_d = normalize_angle_diff(psi_d, psi(i));

		r11 = cos(psi(i));
		r12 = -sin(psi(i));
		r21 = sin(psi(i));
		r22 = cos(psi(i));

		// Calculate coriolis and dampening matrices according to Fossen, 2011 or Stenersen, 2014.
		Cvv(0) = (-M*v(i) + Y_vdot*v(i) + Y_rdot*r(i)) * r(i);
		Cvv(1) = ( M*u(i) - X_udot*u(i)) * r(i);
		Cvv(2) = (( M*v(i) - Y_vdot*v(i) - Y_rdot*r(i) ) * u(i) +
		            ( -M*u(i) + X_udot*u(i)) * v(i));

		Dvv(0) = - (X_u + X_uu*fabs(u(i)) + X_uuu*u(i)*u(i)) * u(i);
		Dvv(1) = - ((Y_v*v(i) + Y_r*r(i)) +
		              (Y_vv*fabs(v(i))*v(i) + Y_vvv*v(i)*v(i)*v(i)));
		Dvv(2) = - ((N_v*v(i) + N_r*r(i)) +
		              (N_rr*fabs(r(i))*r(i) + N_rrr*r(i)*r(i)*r(i)));

		this->updateCtrlInput(u_d, psi_d, i);

		// Integrate system

		x(i+1) = x(i) + DT_*(r11*u(i) + r12*v(i));
		y(i+1) = y(i) + DT_*(r21*u(i) + r22*v(i));
		psi(i+1) = psi(i) + DT_*r(i);
		temp = Minv * (tau - Cvv - Dvv);
		u(i+1) = u(i) + DT_*temp(0);
		v(i+1) = v(i) + DT_*temp(1);
		r(i+1) = r(i) + DT_*temp(2);

		x_ = x(i);
		y_ = y(i);
		psi_ = psi(i);
		u_ = u(i);
		v_ = v(i);
		r_= r(i);

		// Keep yaw within [-PI,PI)
		psi(i+1) = normalize_angle(psi(i+1));

	}
}


void shipModel::updateCtrlInput(double u_d, double psi_d, int i){
	double Fx = Cvv[0] + Dvv[0] + Kp_u*M*(u_d - u(i));
	double Fy = 0.0;

    Fy = (Kp_psi * I_z ) * ((psi_d - psi(i)) - Kd_psi*r(i));
    Fy *= 1.0 / rudder_d;

	// Saturate
	if (Fx < Fx_min)
	  Fx = Fx_min;
	if (Fx > Fx_max)
	  Fx = Fx_max;

	if (Fy < Fy_min)
	  Fy = Fy_min;
	if (Fy > Fy_max)
	  Fy = Fy_max;

	tau[0] = Fx;
	tau[1] = Fy;
	tau[2] = rudder_d * Fy;
}

void shipModel::calculate_position_offsets(){
	os_x = A_ - B_;
	os_y = D_ - C_;
}


double normalize_angle(double angle){

	if(std::isinf(angle)) return angle;

	while(angle <= -PI) angle += 2*PI;

	while (angle > PI) angle -= 2*PI;

	return angle;
}

double normalize_angle_diff(double angle, double angle_ref){
	double new_angle;
	double diff = angle_ref - angle;

	if (std::isinf(angle) || std::isinf(angle_ref)) return angle;

	// Get angle within 2*PI of angle_ref
	if (diff > 0){
		new_angle = angle +(diff - fmod(diff, 2*PI));
	}else{
		new_angle = angle + (diff + fmod(-diff, 2*PI));
	}

	// Get angle on side closest to angle_ref
	diff = angle_ref - new_angle;
	if (diff > PI){
		new_angle += 2*PI;
	}else if (diff < -PI){
		new_angle -= 2*PI;
	}
	return new_angle;
}
