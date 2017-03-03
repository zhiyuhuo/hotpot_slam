#ifndef TRACKER_H_
#define TRACKER_H_

// this file will define a class to compute the relative changes between two neighbo frames
// the tracker will be a static instance which works as a member of the 

#include <iostream>
#include <queue>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include "rgbd_frame.h"
#include "worldmap.h"
#include "common_headers.h"

using namespace std;
using namespace cv;

class CRGBDFrame;
class CCameraPara;
class CMap;
class CMapPoint;
class CTrackResult;

class CTracker
{
public:
	// camera parameters
	CCameraPara _rgbcam;
	CCameraPara _depthcam;
	cv::Mat _Kdepth;
	cv::Mat _Krgb;

	// tracking results. the rotation and translation vector of the camera
	cv::Mat _Rvector;
	cv::Mat _Tvector;

	// two frames of tracking. _frame_cur is the new coming frame and the _frame_lst is the last used frame.
	CKeyframe _frame_new;

	// tracking result
	CTrackResult _trackres;


public:
	CTracker()
	{
		CParameterReader para;
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
  		_Rvector = cv::Mat::zeros(3, 1, CV_64F);
  		_Tvector = cv::Mat::zeros(3, 1, CV_64F);

  		// track result
  		_trackres = CTrackResult();
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
  		_Rvector = cv::Mat::zeros(3, 1, CV_64F);
  		_Tvector = cv::Mat::zeros(3, 1, CV_64F);

  		// track result
  		_trackres = CTrackResult();

	}

	~CTracker()
	{
		_Rvector.release();
		_Tvector.release();
	}

	bool trackKeyframe(const CRGBDFrame& rgbframe_cur, CKeyframe& keyframe)
	{
		double score = trackKeyframe(rgbframe_cur, keyframe, _Rvector, _Tvector);

		if (score < 0)
			return false;

		return true;			
	}

	bool trackKeyframe(const CRGBDFrame& rgbdframe_cur, CKeyframe& keyframe, cv::Mat& Rvector, cv::Mat& Tvector)
	{
		//0. extract the image and key points data from the last/reference frame and the image from the current frame.
		cv::Mat img_color_cur = rgbdframe_cur._img_rgb;
		cv::Mat img_gray_cur;
		cv::cvtColor(img_color_cur, img_gray_cur, CV_BGR2GRAY);
		cv::Mat img_cur = img_gray_cur;

		std::vector<cv::KeyPoint> kps_kf;
		cv::Mat dscp_kf = cv::Mat::zeros(keyframe._keypoints.size(), ORBMATCHTH, CV_8U);	

		//1. Use the input R and T as the initail R and T to reproject the map points to the current frame and see if the 
		//map points are visible. select the points that are visible. (Here in this coding stage I will use all the points)
		for (int i = 0; i < keyframe._keypoints.size(); i++) {
			kps_kf.push_back(keyframe._keypoints[i]._kp);
			keyframe._keypoints[i]._dscp.copyTo(dscp_kf.row(i));
		}

		//2. extract cv::KeyPoint s from the current frame and compute the dscp of the current frame.
		cv::ORB orb(1000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
		std::vector<cv::KeyPoint> kps_cur;
		cv::Mat dscp_cur;
		orb.detect(img_cur, kps_cur);
		orb.compute(img_cur, kps_cur, dscp_cur);

		//3. track the keyframe cv::KeyPoint
		std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(NORM_HAMMING, true);
	    matcher.match(dscp_cur, dscp_kf, matches);
	    cout << "matches num: " << matches.size() << std::endl;
	    if (matches.size() < 50)
			std::cout << "matches too low" << std::endl;

		// 4. filter low score match, only keep the high score matches
		//	  we will have several filters. 1st from score, 2nd from fake optical flow
		//    Here we will compute a fake optical flow and abort all the points which does not match the flow
  		std::vector<cv::DMatch> good_matches;
  		int threshold = ORBMATCHTH;
		for( int i = 0; i < matches.size(); i++ ){
			if( matches[i].distance <= threshold) {
				good_matches.push_back(matches[i]);
			}
		}

		std::cout << "good_matches num: " << good_matches.size() << " " << dscp_kf.rows << " " 
		          << float(good_matches.size()) / float(dscp_kf.rows) << std::endl;
		if (good_matches.size() < 30)
			std::cout << "good_matches too low" << std::endl;

		// 5. compute the R,T use solveRansac. Given the 3D points and their projections on the current image
		//  5.1 get 3D point and 2d pixels
		std::vector<cv::Point3d> p3d;
		std::vector<cv::Point2d> pn_cur;
		for (int i = 0; i < good_matches.size(); i++) {
			if (keyframe._keypoints[good_matches[i].trainIdx].ifTrack()) {
				p3d.push_back( keyframe._keypoints[good_matches[i].trainIdx]._pmappoint->_p3d);
				pn_cur.push_back(kps_cur[good_matches[i].queryIdx].pt);
			}
		}
		
		//  5.3 define input and output parameters
	    cv::Mat rvector;
		cv::Mat tvector;
		cv::Mat inliers;

		//  5.4 calculate r and t // TODO, ransac is needed here!
		solvePNPCustom(p3d, pn_cur, _Krgb, rvector, tvector, inliers);
		std::cout << "inliers num and portion: " << float(inliers.rows) << ", " <<  float(kps_kf.size()) 
		          << ", " << float(inliers.rows) / float(kps_kf.size())  << std::endl;

		// examine if the solution is successful
		if (inliers.rows < 10)
			std::cout << "inliers for PnP too low" << std::endl;

		// rectify the rotation angle and translate to match the camera coordinate
		rvector.copyTo(Rvector);
		tvector.copyTo(Tvector);

		std::cout << "Rvector:\n" << Rvector << std::endl;
		std::cout << "Tvector:\n" << Tvector << std::endl;

		double score = float(inliers.rows) / kps_kf.size();
		std::cout << "final Score: " << std::endl << "     " << score << std::endl;

		///// show the good matches
		cv::namedWindow("matched_key_points");
		cv::Mat img_match = img_color_cur.clone();
		std::vector<cv::KeyPoint> kps_show;
		for (int i = 0; i < good_matches.size(); i++) {
			kps_show.push_back(kps_cur[good_matches[i].queryIdx]);
		}
		cv::drawKeypoints(img_color_cur, kps_show, img_match);
		std::string str_show = std::to_string(kps_kf.size()) + " " + std::to_string(good_matches.size()) + " " + std::to_string(inliers.rows);
		cv::putText(img_match, str_show, Point(10, 400), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(0), 3, 8);
		cv::imshow("matched_key_points", img_match);
		cv::moveWindow("matched_key_points", 1600, 900);
		cv::waitKey(1);

		// update the _frame_new using current frame
		buildNewKeyFrame(keyframe, Rvector, Tvector, kps_cur, dscp_cur, good_matches, _frame_new);
		updateTrackRes(keyframe._keypoints.size(), good_matches.size(), inliers.rows);
		// update the dscp of the current keyframe 

		return 0;		
	}

	/// track the world map and return the score
	double trackWorldMap(const CRGBDFrame& rgbdframe_cur, CMap& worldmap, cv::Mat& Rvector, cv::Mat& Tvector)
	{
		//0. extract the image and key points data from the last/reference frame and the image from the current frame.
		cv::Mat img_color_cur = rgbdframe_cur._img_rgb;
		cv::Mat img_gray_cur;
		cv::cvtColor(img_color_cur, img_gray_cur, CV_BGR2GRAY);
		cv::Mat img_cur = img_gray_cur;

		std::vector<cv::Point3d> p3d_map;
		cv::Mat dscp_map; 	

		//1. Use the input R and T as the initail R and T to reproject the map points to the current frame and see if the 
		//map points are visible. select the points that are visible. (Here in this coding stage I will use all the points)
		std::map<int, CMapPoint*>::const_iterator it_map;
		std::vector<int> valid_idx;
		for (it_map = worldmap._worldpoints.begin(); it_map != worldmap._worldpoints.end(); it_map++) {
			cv::Point2d im_proj = project3Dto2D(it_map->second->_p3d, _Krgb, Rvector, Tvector);
			if (im_proj.x > 10 && im_proj.x < img_color_cur.cols - 10 && im_proj.y > 10 && im_proj.y < img_color_cur.rows) {
				valid_idx.push_back(it_map->first);
				p3d_map.push_back(it_map->second->_p3d);
			}
		}

		dscp_map = cv::Mat::zeros(valid_idx.size(), ORBMATCHTH, CV_8U);
		for (int i = 0; i < valid_idx.size(); i++) {
			worldmap._worldpoints[valid_idx[i]]->_dscp.copyTo(dscp_map.row(i));
		}

		//2. extract key points and compute the dscp of the current frame.
		cv::ORB orb(1000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
		std::vector<cv::KeyPoint> kps_cur;
		cv::Mat dscp_cur;
		orb.detect(img_cur, kps_cur);
		orb.compute(img_cur, kps_cur, dscp_cur);

		//3. track the map points
		std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(NORM_HAMMING, true);
	    matcher.match(dscp_cur, dscp_map, matches);
	    cout << "matches num: " << matches.size() << std::endl;
	    if (matches.size() < 30)
			return -1;

		// 4. filter low score match, only keep the high score matches
		//	  we will have several filters. 1st from score, 2nd from fake optical flow
		//    Here we will compute a fake optical flow and abort all the points which does not match the flow
  		std::vector<cv::DMatch> good_matches;
  		int threshold = 64;
		for( int i = 0; i < matches.size(); i++ ){
			if( matches[i].distance <= threshold) {
				good_matches.push_back(matches[i]);
			}
		}

		std::cout << "good_matches num: " << good_matches.size() << " " << dscp_map.rows << " " 
		          << float(good_matches.size()) / float(dscp_map.rows) << std::endl;
		if (good_matches.size() < 20)
			return -2;

		// 5. compute the R,T use solveRansac. Given the 3D points and their projections on the current image
		//  5.1 get 3D point and 2d pixels
		std::vector<cv::Point3d> p3d;
		std::vector<cv::Point2d> pn_cur;
		for (int i = 0; i < good_matches.size(); i++) {
			p3d.push_back(p3d_map[good_matches[i].trainIdx]);
			pn_cur.push_back(kps_cur[good_matches[i].queryIdx].pt);
		}
		
		//  5.3 define input and output parameters
	    cv::Mat rvector;
		cv::Mat tvector;
		cv::Mat inliers;

		//  5.4 calculate r and t // TODO, ransac is needed here!
		solvePNPCustom(p3d, pn_cur, _Krgb, rvector, tvector, inliers);
		std::cout << "inliers num and portion: " << float(inliers.rows) << ", " <<  float(p3d_map.size()) 
		          << ", " << float(inliers.rows) / float(p3d_map.size())  << std::endl;

		// examine if the solution is successful
		if (inliers.rows < 10)
			return -3;

		// rectify the rotation angle and translate to match the camera coordinate
		rvector.copyTo(Rvector);
		tvector.copyTo(Tvector);

		std::cout << "Rvector:\n" << Rvector << std::endl;
		std::cout << "Tvector:\n" << Tvector << std::endl;

		double score = float(inliers.rows) / p3d_map.size();
		std::cout << "final Score: " << std::endl << "     " << score << std::endl;

		///// show the good matches
		cv::namedWindow("matched_key_points");
		cv::Mat img_match = img_color_cur.clone();
		std::vector<cv::KeyPoint> kps_show;
		for (int i = 0; i < good_matches.size(); i++) {
			kps_show.push_back(kps_cur[good_matches[i].queryIdx]);
		}
		cv::drawKeypoints(img_color_cur, kps_show, img_match);
		std::string str_show = std::to_string(p3d_map.size()) + " " + std::to_string(good_matches.size()) + " " + std::to_string(inliers.rows);
		cv::putText(img_match, str_show, Point(10, 400), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(0), 3, 8);
		cv::imshow("matched_key_points", img_match);
		cv::moveWindow("matched_key_points", 1600, 900);
		cv::waitKey(1);

		updateTrackRes(worldmap._worldpoints.size(), good_matches.size(), inliers.rows);

		return 0;
	}

	void buildNewKeyFrame(CKeyframe keyframe, cv::Mat rvector, cv::Mat tvector, 
						  std::vector<cv::KeyPoint> kps, cv::Mat dscp, std::vector<cv::DMatch> matches, 
		                  CKeyframe& frame_new)
	{
		frame_new.reset();
		frame_new.setRT(rvector, tvector);
		for (int i = 0; i < kps.size(); i++) {
			CKeypoint kp(kps[i], dscp.row(i));
			frame_new.addKeypoint(kp);
		}
		int id_kf, id_new;
		for (int i = 0; i < matches.size(); i++) {
			id_kf = matches[i].trainIdx;
			id_new = matches[i].queryIdx;
			if (keyframe._keypoints[id_kf].ifTrack()) {
				frame_new._keypoints[id_new].trackToMapPoint( keyframe._keypoints[id_kf]._pmappoint );
			}
		}

		// for (int i = 0; i < frame_new->_keypoints.size(); i++) {
		// 	std::cout << i << "- : " << frame_new->_keypoints[i]._kp.pt << frame_new->_keypoints[i]._dscp << std::endl;
		// 	if (frame_new->_keypoints[i].ifTrack()) {
		// 		std::cout << frame_new->_keypoints[i]._pmappoint->_p3d << std::endl;
		// 	}
		// 	else {
		// 		std::cout << "kp not tracked" << endl;
		// 	}
		// }
	}

	void updateTrackRes(int kps_kp, int good_matches, int inliers)
	{
		_trackres.reset();
		_trackres._matchesnum = good_matches;
		_trackres._inliersnum = inliers;
		_trackres._score = _trackres._inliersnum / _trackres._matchesnum;

		if (_trackres._matchesnum < 30 && _trackres._inliersnum < 10)
		{
			_trackres._iftrack = false;
		}

		if (_trackres._matchesnum < 60) {
			_trackres._ifshift = true;
		}

	}
 
 	// customize PNP to overcome the bug in opencv on data type
	void solvePNPOld(std::vector<Point3d> points, std::vector<Point2d> pixels, cv::Mat K, cv::Mat& r, cv::Mat& t, cv::Mat& inliers)
	{
		vector<Point2f> pn;
		vector<Point3f> pts;
		cv::Mat distortionCoefficients;

		for (int i = 0; i < pixels.size(); i++) {
			pn.push_back(Point2f(float(pixels[i].x), float(pixels[i].y)));
			pts.push_back(Point3f(float(points[i].x), float(points[i].y), float(points[i].z)));
		}

		// first iteration
		cv::solvePnPRansac(pts, pn, K, distortionCoefficients, r, t, true, 100, 8.0, 100, inliers, CV_EPNP);

		vector<Point2f> pn_2;
		vector<Point3f> pts_2;
		vector<int> idx_vec;

		for (int i = 0; i < inliers.rows; i++) {
			pn_2.push_back(pn[inliers.at<int>(i, 0)]);
			pts_2.push_back(pts[inliers.at<int>(i, 0)]);
			idx_vec.push_back(inliers.at<int>(i, 0));
		}

		// second iteration
		cv::Mat inliers_2;
		cv::solvePnPRansac(pts_2, pn_2, K, distortionCoefficients, r, t, true, 50, 1.0, 25, inliers_2, CV_EPNP);

		// change the number in inlier2 to the index in inlier1
		for (int i = 0; i < inliers_2.rows; i++) {
			inliers_2.at<int>(i, 0) = inliers.at<int>(inliers_2.at<int>(i, 0), 0);
		}

		inliers_2.copyTo(inliers);
	}

	void solvePNPCustom(std::vector<Point3d> points, std::vector<Point2d> pixels, cv::Mat K, cv::Mat& r, cv::Mat& t, cv::Mat& inliers)
	{
		vector<Point2f> pn;
		vector<Point3f> pts;
		cv::Mat distortionCoefficients;

		for (int i = 0; i < pixels.size(); i++) {
			pn.push_back(Point2f(float(pixels[i].x), float(pixels[i].y)));
			pts.push_back(Point3f(float(points[i].x), float(points[i].y), float(points[i].z)));
		}

		// first iteration
		cv::solvePnPRansac(pts, pn, K, distortionCoefficients, r, t, true, 100, 5.0, 50, inliers, CV_EPNP);
		//cv::solvePnP(pts, pn, K, distortionCoefficients, r, t, true, 100, 8.0, 100, inliers, CV_EPNP);

		vector<Point2f> pn_2;
		vector<Point3f> pts_2;
		vector<int> idx_vec;

		for (int i = 0; i < inliers.rows; i++) {
			pn_2.push_back(pn[inliers.at<int>(i, 0)]);
			pts_2.push_back(pts[inliers.at<int>(i, 0)]);
			idx_vec.push_back(inliers.at<int>(i, 0));
		}
		if (inliers.rows < 10)
			return;

		// second iteration
		//cv::Mat inliers_2;
		cv::solvePnP(pts_2, pn_2, K, distortionCoefficients, r, t, true, CV_ITERATIVE);

		// // change the number in inlier2 to the index in inlier1
		// for (int i = 0; i < inliers_2.rows; i++) {
		// 	inliers_2.at<int>(i, 0) = inliers.at<int>(inliers_2.at<int>(i, 0), 0);
		// }
		// inliers_2.copyTo(inliers);
	}

	// display matched pixels in two images.
	void displayKeypointsMatching(cv::Mat img, std::vector<cv::Point2d> pn_lst, std::vector<cv::Point2d> pn_cur, cv::Mat inliers)
	{
		// Display th corresponding points
		cv::Mat image_show = img.clone();
		for (int i = 0; i < inliers.rows; i++) {
			int id = inliers.at<int>(i, 0);
			cv::circle(image_show, pn_lst[id], 2, cv::Scalar(255, 0, 0), 1, CV_AA, 0);
			cv::circle(image_show, pn_cur[id], 2, cv::Scalar(0, 0, 255), 1, CV_AA, 0);
			cv::line(image_show, pn_lst[id], pn_cur[id], cv::Scalar(0, 255, 0), 1, CV_AA, 0);
		}
		cv::Mat image_ref_show;
		//cv::drawKeypoints (_keyframes[_keyframes.size()-1]._img_rgb, _keyframes[_keyframes.size()-1]._kps, image_ref_show);

		cv::namedWindow("line");
		cv::moveWindow("line", 1600, 900);
		cv::namedWindow("ref");
		cv::moveWindow("ref", 900, 900);
		cv::imshow("line", image_show);
		cv::imshow("ref", image_ref_show);
		cv::waitKey(1);
		image_show.release();
	}

	cv::Point2d project3Dto2D(const cv::Point3d& ptd, const cv::Mat& K, const cv::Mat& rvec, const cv::Mat& tvec) 
	{
		std::vector<cv::Point3f> pt(1, Point3f((float)ptd.x, (float)ptd.y, (float)ptd.z));
		std::vector<cv::Point2f> im(1);
		cv::Mat distortionCoefficients;
		cv::projectPoints(pt, rvec, tvec, K, distortionCoefficients, im);
		cv::Point2d res(im[0].x, im[0].y);
		return res;
	}

};



#endif