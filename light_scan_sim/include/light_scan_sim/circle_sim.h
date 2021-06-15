/**
 * light_scan_sim circle_sim.h
 * @brief Simulate laser rays against circle
 *
 * @copyright 2021 Luman Zhao
 * @author Luman Zhao
 */

#ifndef LIGHT_SCAN_SIM_Circle_SIM_H
#define LIGHT_SCAN_SIM_Circle_SIM_H

#include <Box2D/Box2D.h>
#include <light_scan_sim/Circle.h>
#include <light_scan_sim/CircleList.h>
#include <light_scan_sim/Material.h>
#include <light_scan_sim/MaterialList.h>
#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <random>

class CircleSim
{
private:
  light_scan_sim::CircleList circles_;
  light_scan_sim::MaterialList materials_;

  std::shared_ptr<b2World> world_ = nullptr;

  void InitializeWorld();

public:
  CircleSim(light_scan_sim::CircleList circles, light_scan_sim::MaterialList materials);

  bool Trace(double x, double y, double theta, double length, double ray_max, double &range);
};

#endif
