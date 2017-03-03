#ifndef COMMON_HEADERS_H_
#define COMMON_HEADERS_H_

#define ORBDSCP_L 32
#define ORBMATCHTH 32

#include "parameter_reader.h"
#include "camera_para.h"
#include "rgbd_frame.h"
#include "keypoint.h"
#include "keyframe.h"
#include "track_camera.h"
#include "map_point.h"
#include "worldmap.h"
#include "g2o_bridge.h"
#include "initializer.h"
#include "slam_base.h"


std::vector<double> rotationMatrixToQuanterion(Mat r)
{
  std::vector<double> q(4, 0);
  double S;

  double m00 = r.at<double>(0, 0);
  double m01 = r.at<double>(0, 1);
  double m02 = r.at<double>(0, 2);
  double m10 = r.at<double>(1, 0);
  double m11 = r.at<double>(1, 1);
  double m12 = r.at<double>(1, 2);
  double m20 = r.at<double>(2, 0);
  double m21 = r.at<double>(2, 1);
  double m22 = r.at<double>(2, 2);


  if (m00 > m11 && m00 > m22) {
    S = sqrt( 1.0 + m00 - m11 - m22 ) * 2;
    q[0] = 0.25 / S; 
    q[1] = (m01 + m10 ) / S;
    q[2] = (m02 + m20 ) / S;
    q[3] = (m12 - m21 ) / S; 
  } else if (m11 > m22) {
    S = sqrt( 1.0 + m11 - m00 - m22 ) * 2;
    q[0] = (m01 + m10 ) / S;
    q[1] = 0.25 / S; 
    q[2] = (m12 + m21 ) / S;
    q[3] = (m02 - m20 ) / S; 
  } else {
    S = sqrt( 1.0 + m22 - m00 - m11 ) * 2;
    q[0] = (m02 + m20 ) / S;
    q[1] = (m12 + m21 ) / S;
    q[2] = 0.25 / S; 
    q[3] = (m01 - m10 ) / S; 
  }

  return q;
}

std::vector<double> eulerToQuaternion(double pitch, double roll, double yaw)
{
  std::vector<double> q(4, 0);
  double t0 = std::cos(yaw * 0.5);
  double t1 = std::sin(yaw * 0.5);
  double t2 = std::cos(roll * 0.5);
  double t3 = std::sin(roll * 0.5);
  double t4 = std::cos(pitch * 0.5);
  double t5 = std::sin(pitch * 0.5);

  q[0] = t0 * t2 * t4 + t1 * t3 * t5;
  q[1] = t0 * t3 * t4 - t1 * t2 * t5;
  q[2] = t0 * t2 * t5 + t1 * t3 * t4;
  q[3] = t1 * t2 * t4 - t0 * t3 * t5;
  return q;
}











#endif