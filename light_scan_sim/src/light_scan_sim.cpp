/**
 * light_scan_sim light_scan_sim.cpp
 * @brief Monitor map and tf data, publish simulated laser scan
 *
 * @copyright 2017 Joseph Duchesne
 * @author Joseph Duchesne
 *
 */

#include <math.h>

#include "light_scan_sim/light_scan_sim.h"

/**
 * @brief Initialize light scan sim class
 *
 * @param node The ros node handle
 */
LightScanSim::LightScanSim(ros::NodeHandle node)
{
  // Load settings
  ray_cast_ =
      std::make_shared<RayCast>(node.param<double>("range/min", 1.0), node.param<double>("range/max", 20.0),
                                node.param<double>("angle/min", -M_PI_2), node.param<double>("angle/max", M_PI_2),
                                node.param<double>("angle/increment", 0.01), node.param<double>("range/noise", 0.01));

  node.getParam("map/topic", map_topic_);
  node.getParam("map/materials_topic", materials_topic_);
  node.getParam("map/segments_topic", segments_topic_);
  node.getParam("map/circles_topic", circles_topic_);
  node.getParam("laser/topic", laser_topic_);
  node.getParam("marker/topic", marker_topic_);

  node.getParam("map/image_frame", image_frame_);
  node.getParam("laser/frame", laser_frame_);

  // Subscribe / Publish
  map_sub_ = node.subscribe(map_topic_, 1, &LightScanSim::MapCallback, this);
  materials_sub_ = node.subscribe(materials_topic_, 1, &LightScanSim::MaterialsCallback, this);
  segments_sub_ = node.subscribe(segments_topic_, 1, &LightScanSim::SegmentsCallback, this);
  circles_sub_ = node.subscribe(circles_topic_, 1, &LightScanSim::CirclesCallback, this);
  obstacles_sub_ = node.subscribe("/obstacle_states", 1, &LightScanSim::obstacleCallback, this);

  laser_pub_ = node.advertise<sensor_msgs::LaserScan>(laser_topic_, 1);
  laser_marker_pub_ = node.advertise<visualization_msgs::Marker>(marker_topic_, 1);
}

void LightScanSim::obstacleCallback(const asv_msgs::StateArray::ConstPtr& msg)
{
  // ROS_WARN("light sim obstacle callback");
  obstacles_ = msg->states;
}

/**
 * @brief Recieve the subscribed map and process its data
 *
 * @param grid The map occupancy grid
 */
void LightScanSim::MapCallback(const nav_msgs::OccupancyGrid::Ptr& grid)
{
  map_ = *grid;  // Copy the entire message

  // Convert OccupancyGrid to cv::Mat, uint8_t
  cv::Mat map_mat = cv::Mat(map_.info.height, map_.info.width, CV_8UC1, map_.data.data());
  // Set unknown space (255) to free space (0)
  // 4 = threshold to zero, inverted
  // See: http://docs.opencv.org/3.1.0/db/d8e/tutorial_threshold.html
  cv::threshold(map_mat, map_mat, 254, 255, 4);

  // Update map
  ray_cast_->SetMap(map_mat, map_.info.resolution, map_.info.origin.position.x, map_.info.origin.position.y);

  // Create transform from map tf to image tf
  map_to_image_.setOrigin(
      tf::Vector3(map_.info.origin.position.x, map_.info.origin.position.y, map_.info.origin.position.z));
  // Image is in standard right hand orientation
  map_to_image_.setRotation(tf::createQuaternionFromRPY(0, 0, 0));

  map_loaded_ = true;
  if (map_loaded_ && segments_loaded_ && materials_loaded_)
  {
    ray_cast_->SetSegments(segments_, materials_);
  }

  if (map_loaded_ && circles_loaded_ && materials_loaded_)
  {
    ray_cast_->SetCircles(circles_, materials_);
  }
}

/**
 * @brief Load materials and set segments/materials on ray_cast_ if possible
 *
 * @param materials The material list
 */
void LightScanSim::MaterialsCallback(const light_scan_sim::MaterialList::Ptr& materials)
{
  materials_ = *materials;
  materials_loaded_ = true;

  if (map_loaded_ && segments_loaded_ && materials_loaded_)
  {
    ray_cast_->SetSegments(segments_, materials_);
  }

  if (map_loaded_ && circles_loaded_ && materials_loaded_)
  {
    ray_cast_->SetCircles(circles_, materials_);
  }
}

/**
 * @brief Load segments and set segments/materials on ray_cast_ if possible
 *
 * @param segments The segment list
 */
void LightScanSim::SegmentsCallback(const light_scan_sim::SegmentList::Ptr& segments)
{
  segments_ = *segments;
  segments_loaded_ = true;

  if (map_loaded_ && segments_loaded_ && materials_loaded_)
  {
    ray_cast_->SetSegments(segments_, materials_);
  }
}

/**
 * @brief Load circles and set circles/materials on ray_cast_ if possible
 *
 * @param circles The circles list
 */
void LightScanSim::CirclesCallback(const light_scan_sim::CircleList::Ptr& circles)
{
  ROS_WARN("[LightScanSim] get obstacle circles....");
  circles_ = *circles;
  circles_loaded_ = true;

  if (map_loaded_ && circles_loaded_ && materials_loaded_)
  {
    ray_cast_->SetCircles(circles_, materials_);
  }
}

/**
 * @brief Generate and publish the simulated laser scan
 */
void LightScanSim::Update()
{
  if (!map_loaded_)
  {
    ROS_WARN("LightScanSim: Update called, no map yet");
    return;
  }

  // Broadcast the tf representing the map image
  tf_broadcaster_.sendTransform(
      tf::StampedTransform(map_to_image_, ros::Time::now(), map_.header.frame_id, image_frame_));

  // Use that transform to generate a point in image space
  tf::StampedTransform image_to_laser, map_to_laser;
  try
  {
    // tf_listener_.waitForTransform(image_frame_, laser_frame_, ros::Time(0), ros::Duration(10.));
    tf_listener_.lookupTransform(image_frame_, laser_frame_, ros::Time(0), image_to_laser);
    tf_listener_.lookupTransform(map_.header.frame_id, laser_frame_, ros::Time(0), map_to_laser);
  }
  catch (tf::TransformException& ex)
  {
    // ROS_WARN("LightScanSim: %s",ex.what());
    return;
  }

  // Convert that point from m to px
  cv::Point laser_point(image_to_laser.getOrigin().x() / map_.info.resolution,
                        image_to_laser.getOrigin().y() / map_.info.resolution);
  // And get the yaw
  double roll, pitch, yaw;
  image_to_laser.getBasis().getRPY(roll, pitch, yaw);

  // Generate the ray cast laser scan at that point and orientation
  sensor_msgs::LaserScan scan = ray_cast_->Scan(laser_point, yaw);

  // Set the header values
  scan.header.stamp = image_to_laser.stamp_;  // Use correct time
  scan.header.frame_id = laser_frame_;        // set laser's tf

  double start_roll, start_pitch, start_yaw;
  map_to_laser.getBasis().getRPY(start_roll, start_pitch, start_yaw);

  geometry_msgs::Point start_point;
  start_point.x = map_to_laser.getOrigin().x();
  start_point.y = map_to_laser.getOrigin().y();
  start_point.z = map_to_laser.getOrigin().z();

  // update the laser scan with other obstacle ships
  updateLaserWithObstacles(start_point, start_yaw, scan);
  publishLaserMarker(start_point, start_yaw, scan);
  // And publish the laser scan
  // ROS_WARN_STREAM("laser scan sizes: " << scan.ranges.size());
  laser_pub_.publish(scan);
}

void LightScanSim::updateLaserWithObstacles(const geometry_msgs::Point& start_point, const double& yaw,
                                            sensor_msgs::LaserScan& scan)
{
  if (obstacles_.size() < 1)
    return;

  for (size_t i = 0; i < obstacles_.size(); ++i)
  {
    size_t k = 0;
    geometry_msgs::Point point;
    for (double theta = scan.angle_min; theta <= scan.angle_max; theta += scan.angle_increment)
    {
      point.x = start_point.x + scan.ranges[k] * std::cos(yaw + theta);
      point.y = start_point.y + scan.ranges[k] * std::sin(yaw + theta);
      point.z = start_point.z;

      float e2s_x = point.x - start_point.x;
      float e2s_y = point.y - start_point.y;
      float s2o_x = start_point.x - obstacles_[i].x;
      float s2o_y = start_point.y - obstacles_[i].y;

      float a = e2s_x * e2s_x + e2s_y * e2s_y;
      float b = 2 * (e2s_x * s2o_x + e2s_y * s2o_y);
      float c = s2o_x * s2o_x + s2o_y * s2o_y - 10. * 10.;
      float delta = b * b - 4 * a * c;

      if (delta >= 0)
      {
        delta = std::sqrt(delta);
        float res1 = (-b - delta) / (2 * a);
        float res2 = (-b + delta) / (2 * a);

        if (res1 + TOLERANCE >= 0.0 && res1 < 1.0)
          // point.x = start_point.x + res1 * e2s_x;
          // point.y = start_point.y + res1 * e2s_y;
          scan.ranges[k] = std::hypot(res1 * e2s_x, res1 * e2s_y);
      }

      k++;
    }
  }
}

void LightScanSim::publishLaserMarker(const geometry_msgs::Point& start_point, const double& yaw,
                                      const sensor_msgs::LaserScan& scan)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.header.stamp = ros::Time::now();
  marker.ns = "laserscan";
  marker.action = visualization_msgs::Marker::ADD;
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.scale.x = 1.;
  marker.color.r = 255.0 / 255.;
  marker.color.g = 169. / 255.;
  marker.color.b = 0.0 / 255.0;
  marker.color.a = 1.0;

  size_t i = 0;
  geometry_msgs::Point point;
  for (double a = scan.angle_min; a <= scan.angle_max; a += scan.angle_increment)
  {
    marker.points.push_back(start_point);  // center of a gent

    point.x = start_point.x + scan.ranges[i] * std::cos(yaw + a);
    point.y = start_point.y + scan.ranges[i] * std::sin(yaw + a);
    point.z = start_point.z;
    marker.points.push_back(point);
    i++;
  }
  // ROS_INFO_STREAM("i: " << i);

  laser_marker_pub_.publish(marker);
}
