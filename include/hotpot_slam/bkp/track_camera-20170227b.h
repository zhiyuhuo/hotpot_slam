#ifndef TRACKER_H_
#define TRACKER_H_

// this file will define a class to compute the relative changes between two neighbo frames
// the tracker will be a static instance which works as a member of the 

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include "rgbd_frame.h"
#include "worldmap.h"

using namespace std;
using namespace cv;

class CRGBDFrame;
class CCameraPara;
class CMap;

class CTracker
{
public:
	CCameraPara _rgbcam;
	CCameraPara _depthcam;

	cv::Mat _Kdepth;
	cv::Mat _Krgb;
	cv::Mat _R;
	cv::Mat _T;

	std::vector<CRGBDFrame> _key_frames;
	CRGBDFrame _frame_fst;
	CRGBDFrame _frame_lst;
	CRGBDFrame _frame_cur;

	int _count_running;

public:
	CTracker()
	{
		_count_running = 0;
	}

	CTracker(CParameterReader para)
	{
  		// init rgb camera
  		_rgbcam = CCameraPara(	para.rgb_camera_width, para.rgb_camera_height,
  								para.rgb_camera_fx, para.rgb_camera_fy, para.rgb_camera_cx, para.rgb_camera_cy);
  		double kinect_rgb_camera_intrinsic_para[3][3] =  { { _rgbcam._fx,           0, _rgbcam._cx},
								  						   {           0, _rgbcam._fy, _rgbcam._cy},
								  						   {           0,	   	    0,           1} }; 
  		_Krgb = cv::Mat(3, 3, CV_64F);
  		for (int i = 0; i < _Krgb.rows; i++)
  			for (int j = 0; j < _Krgb.cols; j++)
  				_Krgb.at<double>(i, j) = kinect_rgb_camera_intrinsic_para[i][j];

		// init depth camera
		_depthcam = CCameraPara(para.depth_camera_width, para.depth_camera_height,
								para.depth_camera_fx, para.depth_camera_fy, para.depth_camera_cx, para.depth_camera_cy);
  		double kinect_depth_camera_intrinsic_para[3][3] =  { { _depthcam._fx,              0, _depthcam._cx},
								  							 {             0,  _depthcam._fy, _depthcam._cy},
								  							 {             0,		       0,	  	      1} }; 
  		_Kdepth = cv::Mat(3, 3, CV_64F);
  		for (int i = 0; i < _Kdepth.rows; i++)
  			for (int j = 0; j < _Kdepth.cols; j++)
  				_Kdepth.at<double>(i, j) = kinect_depth_camera_intrinsic_para[i][j];

  		// init R and T 
  		_R = cv::Mat::zeros(3, 1, CV_64F);
  		_T = cv::Mat::zeros(3, 1, CV_64F);

	}

	~CTracker()
	{
		_R.release();
		_T.release();
	}

	int addKeyFrame(const CRGBDFrame& frame) {
		_key_frames.push_back(frame);
		return 0;
	}

	int trackWorldMap(const CRGBDFrame& frame_cur, CMap& worldmap)
	{
		int res = trackRGBFromMWorldMap(frame_cur._img_rgb, worldmap);
		return res;
	}

	int trackRGBFromMWorldMap(const cv::Mat& img_color_cur, CMap& worldmap) 
	{
		// 0. build the vector of mappoints. this include a vector of 
		//    3D points set and a Mat descriptor
		std::vector<int> map_pts_id(worldmap._worldpoints.size());
		std::vector<cv::Point3f> map_pts_3d(worldmap._worldpoints.size()); // this is a fucking OPENCV bug. This animals forgot to add the support for double format.
		cv::Mat map_pts_dscp(worldmap._worldpoints.size(),32,CV_8U); // 32 is the number of 8-bit data unit in a descriptor
		int count = 0;
		std::map<int, CMapPoint>::const_iterator it;
		for (it = worldmap._worldpoints.begin(); it != worldmap._worldpoints.end(); it++) {
			map_pts_id[count] = it->first;
			map_pts_3d[count] = Point3f((float)it->second._p3d.x, (float)it->second._p3d.y, (float)it->second._p3d.z);
			it->second._dscp.copyTo(map_pts_dscp.row(count));
			count++;
		}

		// 1. convert the current/query image to grayscale, smooth it
		cv::Mat img_gray_cur;
		cv::cvtColor(img_color_cur, img_gray_cur, CV_BGR2GRAY);
		cv::Mat img_cur = img_gray_cur;
		//cv::medianBlur(img_cur, img_cur, 3);

		// 2. detect key points on the current image, extract the ORB features and compute descriptor for the current image
		cv::ORB orb(1000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
		std::vector<cv::KeyPoint> kps_cur;
		cv::Mat dscp_cur;
		orb.detect(img_cur, kps_cur);
		orb.compute(img_cur, kps_cur, dscp_cur);

		// 3. match the query frame to the map points, get the matching result.
		std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
	    matcher.match(dscp_cur, map_pts_dscp, matches);
	    if (matches.size() < 10)
			return -1;

		std::cout << kps_cur.size() << ", " << map_pts_3d.size() << ", " << matches.size() << std::endl;

		// 4. reserve the high score matching
		int match_threshold = 64;
		std::vector<DMatch> good_matches;
		for (int i = 0; i < matches.size(); i++) {
			if (matches[i].distance < match_threshold) {
				good_matches.push_back(matches[i]);
			}
		}
		std::cout << "good_matches.size(): " << good_matches.size() << std::endl;
		if (good_matches.size() < 10)
			return -1;

		// 5. extract the reserved matched pixels in the query image, and the 3d positions of them,
		// ready for solvePNP
		std::vector<cv::Point2f> pixels;
		std::vector<cv::Point3f> p3d;
		std::vector<int> indices;
		std::vector<int> dscp_lines;
		for (int i = 0; i < good_matches.size(); i++) {
			cv::Point2f imq = kps_cur[good_matches[i].queryIdx].pt;
			cv::Point3f p3dt = map_pts_3d[good_matches[i].trainIdx];
			int point_id = map_pts_id[good_matches[i].trainIdx];

			pixels.push_back(imq);
			p3d.push_back(p3dt);
			indices.push_back(point_id);
			dscp_lines.push_back(good_matches[i].queryIdx);
		}

		// 6. use official opencv solve PNP to solve the problem
		cv::Mat distortionCoefficients;
	    cv::Mat rvector;
		cv::Mat tvector;
		cv::Mat inliers;
		cv::solvePnPRansac(p3d, pixels, _Krgb, distortionCoefficients, rvector, tvector, 
		               true, 200, 3.0, 100, inliers, CV_EPNP);

		// check if there is valid results
		if (inliers.rows < 100)
			return -2;

		std::string str = std::to_string(inliers.rows) + " " + std::to_string(norm(tvector));
		displayTrackedMapPoints(img_color_cur, pixels, inliers, str);

		std::cout << float(inliers.rows) << ", " <<  float(pixels.size()) << ", " << float(inliers.rows) / float(pixels.size()) << std::endl;

		cv::Mat rmatrix;
		eulerAnglesToRotationMatrix(rvector.at<double>(0,0), rvector.at<double>(1,0), rvector.at<double>(2,0), rmatrix);
		std::cout << rmatrix << std::endl;
		cv::Mat tmatrix = tvector;

		// rectify the rotation angle and translate to match the camera coordinate
		rvector.copyTo(_R);
		tvector.copyTo(_T);

		std::cout << "_R:\n" << _R << std::endl;
		std::cout << "_T:\n" << _T << std::endl;
 
		map_pts_dscp.release();
	
		return 0;
	}

	/// triangulate points
  	void triangulatePixels(const std::vector<cv::Point2d>& im0, const std::vector<cv::Point2d>& im1, 
  							 const cv::Mat& K, const cv::Mat& R0, const cv::Mat& T0, const cv::Mat& R1, const cv::Mat& T1, 
  		                     std::vector<cv::Point3d>& pointcloud)
	{
  		// triangulate using official
		cv::Mat RT0(3,4,CV_64F);
		R0.copyTo(RT0(cv::Rect(0, 0, 3, 3)));
		T0.copyTo(RT0(cv::Rect(3, 0, 1, 3)));
		cv::Mat RT1(3,4,CV_64F);
		R1.copyTo(RT1(cv::Rect(0, 0, 3, 3)));
		T1.copyTo(RT1(cv::Rect(3, 0, 1, 3)));
		cv::Mat pc4dmat(4,im0.size(),CV_64FC4);
  		triangulatePoints(_Krgb*RT0, _Krgb*RT1, im0, im1, pc4dmat);
		pointcloud.resize(im0.size());

  		for (int i = 0; i < pointcloud.size(); i++) {
  			pointcloud[i].x = pc4dmat.at<double>(0,i) / pc4dmat.at<double>(3,i);
  			pointcloud[i].y = pc4dmat.at<double>(1,i) / pc4dmat.at<double>(3,i);
  			pointcloud[i].z = pc4dmat.at<double>(2,i) / pc4dmat.at<double>(3,i);
  		}

  		RT0.release();
  		RT1.release();
	}

	void displayTrackedMapPoints(cv::Mat img, std::vector<cv::Point2f> im, const cv::Mat& inliers, std::string str = "n/a")
	{
		namedWindow("tracked_map_points");
		moveWindow("tracked_map_points", 1600,900);
		// Display th corresponding points
		cv::Mat image_show = img.clone();
		for (int i = 0; i < im.size(); i++) {
			cv::circle(image_show, im[i], 2, cv::Scalar(0, 255, 0), 1, CV_AA, 0);
		}
		for (int i = 0; i < inliers.rows; i++) {
			int id = inliers.at<int>(i, 0);
			cv::circle(image_show, im[id], 3, cv::Scalar(255, 0, 0), 1, CV_AA, 0);
		}
		cv::putText(image_show, str, Point(10, 400), cv::FONT_HERSHEY_COMPLEX_SMALL, 2,  Scalar(0,0,255,255), 2);
		cv::imshow("tracked_map_points", image_show);
		std::string filename = "/home/rokid/Pictures/save_imgs/" + std::to_string(_count_running++) + ".png";
		cv::imwrite(filename.c_str(), image_show);
		cv::waitKey(1);
		image_show.release();		
	}

	cv::Mat eulerAnglesToRotationMatrix(double x, double y, double z, cv::Mat& R)
	{
	    // Calculate rotation about x axis
	    cv::Mat R_x = (Mat_<double>(3,3) <<
	               1,       0,              0,
	               0,       cos(x),   -sin(x),
	               0,       sin(x),   cos(x)
	               );
	     
	    // Calculate rotation about y axis
	    cv::Mat R_y = (Mat_<double>(3,3) <<
	               cos(y),    0,      sin(y),
	               0,               1,      0,
	               -sin(y),   0,      cos(y)
	               );
	     
	    // Calculate rotation about z axis
	    cv::Mat R_z = (Mat_<double>(3,3) <<
	               cos(z),    -sin(z),      0,
	               sin(z),    cos(z),       0,
	               0,               0,                  1);
	     
	     
	    // Combined rotation matrix
	    cv::Mat Rm = R_z * R_y * R_x;
	    Rm.copyTo(R);
	    return R;
	 
	}
};



#endif