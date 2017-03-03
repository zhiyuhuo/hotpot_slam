#ifndef COMMON_HEADERS_H_
#define COMMON_HEADERS_H_

#define ORBDSCP_L 32
#define ORBMATCHTH 32

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class CTrackResult
{
public:
  double _score;
  bool  _iftrack;
  bool  _ifshift;
  double   _matchesnum;
  double   _inliersnum;

  CTrackResult ()
  {
    _score = 1.0;
    _iftrack = true;
    _ifshift = false;
    _matchesnum = 0.0;
    _inliersnum = 0.0;
  }
  ~CTrackResult()
  {}

  void reset()
  {
     _score = 1.0;
    _iftrack = true;
    _ifshift = false;
    _matchesnum = 0.0;
    _inliersnum = 0.0;   
  }
};

cv::Mat eulerAnglesToRotationMatrix(double x, double y, double z, cv::Mat& R)
{
      // Calculate rotation about x axis
      cv::Mat R_x = (cv::Mat_<double>(3,3) <<
                 1,       0,              0,
                 0,       cos(x),   -sin(x),
                 0,       sin(x),   cos(x)
                 );
       
      // Calculate rotation about y axis
      cv::Mat R_y = (cv::Mat_<double>(3,3) <<
                 cos(y),    0,      sin(y),
                 0,               1,      0,
                 -sin(y),   0,      cos(y)
                 );
       
      // Calculate rotation about z axis
      cv::Mat R_z = (cv::Mat_<double>(3,3) <<
                 cos(z),    -sin(z),      0,
                 sin(z),    cos(z),       0,
                 0,               0,                  1);
         
         
      // Combined rotation matrix
      cv::Mat Rm = R_z * R_y * R_x;
      Rm.copyTo(R);
      return R;
     
  }


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









#endif