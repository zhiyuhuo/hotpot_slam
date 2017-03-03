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
	CRGBDFrame _frame_ref;
	CRGBDFrame _frame_cur;
	CKeyframe* _frame_kf;

	// a list of the candidate frames which can to be add to the world map. 
	vector<CRGBDFrame> _frames;

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

	}

	~CTracker()
	{
		_Rvector.release();
		_Tvector.release();
	}

	/// this is a function that track the R,T movement frame by frame and renew the R,T gradually. 
	/// The last key frame and the key points, 3d points are updated frame by frame.
	bool trackIter(CRGBDFrame& frame_cur)
	{
		// Steps:
		// (1) import the current frame (only)
		// (2) track the last frame, get the key points
		// (3) update the last frame, renew the image, keypoints, kps_3d, descriptor
		// only renew the data before optimizing matching !!!
		double score = trackReferenceFrame(frame_cur, _frame_ref, _Rvector, _Tvector);

		if (score < 0)
			return false;

		return true;
	}

	bool trackMap(const CRGBDFrame& frame_cur, CMap& worldmap)
	{
		//double score = trackWorldMap(frame_cur, worldmap, _Rvector, _Tvector);

		_frame_kf = worldmap._keyframes[0];
		double score = trackKeyframe(frame_cur, _frame_kf, _Rvector, _Tvector);

		if (score < 0)
			return false;

		return true;		
	}

	bool trackKeyframe(const CRGBDFrame& frame_cur, CKeyframe* keyframe, cv::Mat& Rvector, cv::Mat& Tvector)
	{
		//0. extract the image and key points data from the last/reference frame and the image from the current frame.
		cv::Mat img_color_cur = frame_cur._img_rgb;
		cv::Mat img_gray_cur;
		cv::cvtColor(img_color_cur, img_gray_cur, CV_BGR2GRAY);
		cv::Mat img_cur = img_gray_cur;

		std::vector<cv::KeyPoint> kps_kf;
		std::vector<bool> ifckeck_kf;
		cv::Mat dscp_kf = cv::Mat::zeros(keyframe->_keypoints.size(), 32, CV_8U);	

		//1. Use the input R and T as the initail R and T to reproject the map points to the current frame and see if the 
		//map points are visible. select the points that are visible. (Here in this coding stage I will use all the points)
		for (int i = 0; i < keyframe->_keypoints.size(); i++) {
			kps_kf.push_back(keyframe->_keypoints[i]._kp);
			ifckeck_kf.push_back(keyframe->_keypoints[i].ifCheck());
			keyframe->_keypoints[i]._dscp.copyTo(dscp_kf.row(i));
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
	    if (matches.size() < 30)
			return -1;

		// 4. filter low score match, only keep the high score matches
		//	  we will have several filters. 1st from score, 2nd from fake optical flow
		//    Here we will compute a fake optical flow and abort all the points which does not match the flow
  		std::vector<cv::DMatch> good_matches;
  		int threshold = 32;
		for( int i = 0; i < matches.size(); i++ ){
			if( matches[i].distance <= threshold) {
				good_matches.push_back(matches[i]);
			}
		}

		std::cout << "good_matches num: " << good_matches.size() << " " << dscp_kf.rows << " " 
		          << float(good_matches.size()) / float(dscp_kf.rows) << std::endl;
		if (good_matches.size() < 20)
			return -2;

		// 5. compute the R,T use solveRansac. Given the 3D points and their projections on the current image
		//  5.1 get 3D point and 2d pixels
		std::vector<cv::Point3d> p3d;
		std::vector<cv::Point2d> pn_cur;
		for (int i = 0; i < good_matches.size(); i++) {
			if (ifckeck_kf[good_matches[i].trainIdx]) {
				p3d.push_back( keyframe->_keypoints[good_matches[i].trainIdx]._pmappoint->_p3d);
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
			return -3;

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

		return 0;		
	}

	void setReferenceFrame(const CRGBDFrame& frame) {
		_frame_ref = frame;
	}

	/// track the camera from the new coming picture to the reference frame (given key points, 3d point positions and ORB descriptor)
	/// output rotation and translation matrix
	/// Attention! it looks that the "const cv::Mat& img_color_lst" parameter is useless
	double trackReferenceFrame(CRGBDFrame& frame_cur, const CRGBDFrame& frame_lst, cv::Mat& Rvector, cv::Mat& Tvector)
	{
		//0. extract the image and key points data from the last/reference frame and the image from the current frame.
		cv::Mat img_color_cur = frame_cur._img_rgb;
		cv::Mat img_color_lst = frame_lst._img_rgb;
		std::vector<KeyPoint> kps_lst = frame_lst._kps;
		std::vector<cv::Point3d> p3d_lst = frame_lst._p3d;
		cv::Mat dscp_lst = frame_lst._dscp;

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

		// 3. match two frames, get the matching 
		std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(NORM_HAMMING, true);
	    matcher.match(dscp_cur, dscp_lst, matches);
	    cout << "matches num: " << matches.size() << std::endl;
	    if (matches.size() < 30)
			return -1;

		// 4. filter low score match, only keep the high score matches
		//	  we will have several filters. 1st from score, 2nd from fake optical flow
		//    Here we will compute a fake optical flow and abort all the points which does not match the flow
  		std::vector<cv::DMatch> good_matches;
  		int threshold = 32;
		for( int i = 0; i < matches.size(); i++ ){
			if( matches[i].distance <= threshold) {
				good_matches.push_back(matches[i]);
			}
		}

		std::cout << "good_matches num: " << good_matches.size() << " " << kps_lst.size() << " " 
		          << float(good_matches.size()) / float(kps_lst.size()) << std::endl;
		if (good_matches.size() < 20)
			return -2;	

		// 5. compute the R,T use solveRansac. Given the 3D points and their projections on the current image
		//  5.1 get 3D point
		std::vector<cv::Point3d> p3d;
		for (int i = 0; i < good_matches.size(); i++) {
			p3d.push_back(p3d_lst[good_matches[i].trainIdx]);
		}

		//  5.2 get two 2d image pixels. we will only use the pixels of the current image
		std::vector<cv::Point2d> pn_lst;
		std::vector<cv::Point2d> pn_cur;
		std::vector<int> idpn_lst;
		std::vector<int> idpn_cur;
		keyPointPairsToPointVectorPairs(kps_lst, kps_cur, good_matches, pn_lst, pn_cur, idpn_lst, idpn_cur);

		///// show the good matches
		cv::namedWindow("good_matches");
		cv::Mat img_match;
		cv::drawMatches(img_color_cur, kps_cur, img_color_lst, kps_lst, good_matches, img_match);
		cv::putText(img_match, std::to_string(good_matches.size()), Point(10, 400), FONT_HERSHEY_SCRIPT_SIMPLEX, 2, Scalar::all(255), 3, 8);
		cv::imshow("good_matches", img_match);
		cv::moveWindow("good_matches", 700, 900);
		cv::waitKey(1);
		
		//  5.3 define input and output parameters
	    cv::Mat rvector;
		cv::Mat tvector;
		cv::Mat inliers;

		// for (int i = 0; i < p3d.size(); i++) {
		// 	std::cout << p3d[i] << pn_cur[i] << std::endl;
		// }

		//  5.4 calculate r and t // TODO, ransac is needed here!
		solvePNP(p3d, pn_cur, _Krgb, rvector, tvector, inliers);
		std::cout << "inliers num and portion: " << float(inliers.rows) << ", " <<  float(pn_lst.size()) 
		          << ", " << float(inliers.rows) / float(pn_lst.size())  << std::endl;

		// examine if the solution is successful
		if (inliers.rows < 10)
			return -3;

		// rectify the rotation angle and translate to match the camera coordinate
		rvector.copyTo(Rvector);
		tvector.copyTo(Tvector);

		std::cout << "Rvector:\n" << Rvector << std::endl;
		std::cout << "Tvector:\n" << Tvector << std::endl;

		double score = float(inliers.rows) / kps_lst.size();
		std::cout << "final Score: " << std::endl << "     " << score << std::endl;


		// check if the current frame should be updated and save to somewhere.
		if (good_matches.size() < kps_lst.size() * 0.5) {
			updateFrame(frame_cur, kps_cur, dscp_cur, good_matches, p3d, rvector, tvector);
			_frames.push_back(frame_cur);
			std::cout << "update the current frame. _______________" << _frames.size() << std::endl;
			return score;
		}

	    return score;
	}

	/// track the world map and return the score
	double trackWorldMap(const CRGBDFrame& frame_cur, CMap& worldmap, cv::Mat& Rvector, cv::Mat& Tvector)
	{
		//0. extract the image and key points data from the last/reference frame and the image from the current frame.
		cv::Mat img_color_cur = frame_cur._img_rgb;
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

		dscp_map = cv::Mat::zeros(valid_idx.size(), 32, CV_8U);
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

		return 0;
	}

	void updateFrame(CRGBDFrame& frame, std::vector<cv::KeyPoint> kps, cv::Mat dscp, 
					 std::vector<DMatch> matches, std::vector<cv::Point3d> p3d, 
		             cv::Mat rvector, cv::Mat tvector) 
	{
		frame._matches_to_ref = matches;
		rvector.copyTo(frame._Rvector);
		tvector.copyTo(frame._Tvector);
		frame._p3d = p3d;
		std::vector<cv::KeyPoint> kps_new(frame._matches_to_ref.size());
		Mat dscp_new(matches.size(), 32, CV_8U);
		for (int i = 0; i < frame._matches_to_ref.size(); i++) {
			kps_new[i] = kps[frame._matches_to_ref[i].queryIdx];
			dscp.row(frame._matches_to_ref[i].queryIdx).copyTo(dscp_new.row(i));
		}
		frame._p3d = p3d;
		frame._kps = kps_new;
		dscp_new.copyTo(frame._dscp);

	}

	// This function is used to triangulate points from two frames and one reference key frame.
	// we will select points with large shift in pixel, close and middle range and low reprojection error.
	CRGBDFrame triangulateNew3DPointsFromWaitingFramesAndReferenceFrame(CRGBDFrame f0, CRGBDFrame f1, CRGBDFrame frame_ref) 
	{
		//find the good new matches of the f0 and f1 two frames. 
		std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(NORM_HAMMING, true);
	    matcher.match(f1._dscp, f0._dscp, matches); // query, train

	    std::vector<cv::DMatch> new_matches;
	    int id0, id1;
	    int count = 0;
	    for (int i = 0; i < matches.size(); i++) {
	    	bool if_skip = false;
	    	id0 = matches[i].trainIdx;
	    	id1 = matches[i].queryIdx;
	    	for (int j = 0; j < f0._matches_to_ref.size(); j++) {
	    		if ( id0 == f0._matches_to_ref[j].queryIdx) {
		    		if_skip = true;
		    		break;	    			
	    		}
	    	}
	    	for (int j = 0; j < f1._matches_to_ref.size(); j++) {
	    		if ( id1 == f1._matches_to_ref[j].queryIdx) {
		    		if_skip = true;
		    		break;	    			
	    		}
	    	}
	    	if (if_skip) {
	    		count++;
	    		continue;
	    	}
	    	else {
	    		new_matches.push_back(matches[i]);
	    	}
	    }

	    int threshold = 16;
	    std::vector<cv::DMatch> good_new_matches;
	    for (int i = 0; i < new_matches.size(); i++) {
	    	if (new_matches[i].distance < threshold) {
	    		good_new_matches.push_back(new_matches[i]);
	    	}
	    }

	    // triangulation
		std::vector<cv::Point2d> pn0, pn1;
		cv::Mat R0, T0, R1, T1;
		std::vector<cv::Point3d> triang_points;
		for (int i = 0; i < good_new_matches.size(); i++) {
			pn0.push_back(f0._kps[good_new_matches[i].trainIdx].pt);
			pn1.push_back(f1._kps[good_new_matches[i].queryIdx].pt);
		}
		eulerAnglesToRotationMatrix(f0._Rvector.at<double>(0,0), f0._Rvector.at<double>(1,0), f0._Rvector.at<double>(2,0), R0);
		eulerAnglesToRotationMatrix(f1._Rvector.at<double>(0,0), f1._Rvector.at<double>(1,0), f1._Rvector.at<double>(2,0), R1);
		T0 = f0._Tvector;
		T1 = f1._Tvector;
		//std::cout << R0 << T0 << R1 << T1 << std::endl;

  		triangulatePixels(pn0, pn1, _Krgb, R0, T0, R1, T1, triang_points);
  		for (int i = 0; i < good_new_matches.size(); i++) {
  			//std::cout << i << ": " << pn0[i] << " " << pn1[i] << " " << good_new_matches[i].distance << "\t" << triang_points[i] << std::endl;
  		}

  		// filter points 
  		std::vector<DMatch> points_get_matches;
  		std::vector<cv::Point3d> new_points;
  		std::vector<double> er = computeReProjectionError(pn0, pn1, _Krgb, f0._Rvector, f0._Tvector, f1._Rvector, f1._Tvector, triang_points);
  		for (int i = 0; i < good_new_matches.size(); i++) {

  			if (triang_points[i].z > 0.5 && triang_points[i].z < 5.0 && er[i] < 1.0 && cv::norm(pn1[i]-pn0[i]) > 0 ) {
  				std::cout << i << ": " << triang_points[i].z << ", " << cv::norm(pn1[i]-pn0[i]) << ", " << er[i] << std::endl;
  				points_get_matches.push_back(good_new_matches[i]);
  				new_points.push_back(triang_points[i]);
  			}
  		}

  		// build a new key frame;
  		CRGBDFrame new_key_frame;
  		std::vector<cv::KeyPoint> kps_new;
  		std::vector<cv::Point3d> pts_new;
  		Mat dscp_new(f1._matches_to_ref.size() + points_get_matches.size(), 32, CV_8U);
  		f1._img_rgb.copyTo(new_key_frame._img_rgb);
  		// insert key points that have 3d points to ref frame
  		int ct = 0;
  		for (int i = 0; i < f1._matches_to_ref.size(); i++) {
  			kps_new.push_back(f1._kps[f1._matches_to_ref[i].queryIdx]);
  			pts_new.push_back(frame_ref._p3d[f1._matches_to_ref[i].trainIdx]);
  			f1._dscp.row(f1._matches_to_ref[i].queryIdx).copyTo(dscp_new.row(ct));
  			ct++;
  		}

  		/*
  		// insert new triangulated key points
  		for (int i = 0; i < points_get_matches.size(); i++) {
  			kps_new.push_back(f1._kps[points_get_matches[i].queryIdx]);
  			pts_new.push_back(new_points[i]);
  			f1._dscp.row(points_get_matches[i].queryIdx).copyTo(dscp_new.row(ct));
  			ct++;
  		}
		*/

  		new_key_frame._kps = kps_new;
  		new_key_frame._p3d = pts_new;
  		dscp_new.copyTo(new_key_frame._dscp);
  		

  		return new_key_frame;
	}

	/// triangulate points
  	void triangulatePixels(const std::vector<cv::Point2d>& pn0, const std::vector<cv::Point2d>& pn1, 
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
		cv::Mat pc4dmat(4,pn0.size(),CV_64FC4);
  		triangulatePoints(_Krgb*RT0, _Krgb*RT1, pn0, pn1, pc4dmat);
		pointcloud.resize(pn0.size());

  		for (int i = 0; i < pointcloud.size(); i++) {
  			pointcloud[i].x = pc4dmat.at<double>(0,i) / pc4dmat.at<double>(3,i);
  			pointcloud[i].y = pc4dmat.at<double>(1,i) / pc4dmat.at<double>(3,i);
  			pointcloud[i].z = pc4dmat.at<double>(2,i) / pc4dmat.at<double>(3,i);

  			if (pointcloud[i].z < 0) {
  				pointcloud[i].x *= -1.0;
  				pointcloud[i].y *= -1.0;
  				pointcloud[i].z *= -1.0;
  			}
  		}

  		RT0.release();
  		RT1.release();
	}
 
 	// customize PNP to overcome the bug in opencv on data type
	void solvePNP(std::vector<Point3d> points, std::vector<Point2d> pixels, cv::Mat K, cv::Mat& r, cv::Mat& t, cv::Mat& inliers)
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

    /// get the cv::Point2d type key points pair list from two key points set with their matchings
    /// then the two lists of points are in the same length
	void keyPointPairsToPointVectorPairs(std::vector<cv::KeyPoint> kpt, std::vector<cv::KeyPoint> kpq, std::vector<DMatch> m, 
		                                 std::vector<cv::Point2d>& pst, std::vector<cv::Point2d>& psq,
		                                 std::vector<int>& idxt, std::vector<int>& idxq)
	{
		for (int i = 0; i < m.size(); i++) {
			cv::Point2d pt, pq;
		  	pt = kpt[m[i].trainIdx].pt;
		  	pq = kpq[m[i].queryIdx].pt;

		  	pst.push_back(pt);
		  	idxt.push_back(m[i].trainIdx);

		  	psq.push_back(pq);
		  	idxq.push_back(m[i].queryIdx);
	  	}
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

	// compute the reprojection a 3d point cloud to two 2d images 
	std::vector<double> computeReProjectionError(const std::vector<cv::Point2d>& pn0, const std::vector<cv::Point2d>& pn1, 
  							 const cv::Mat& K, const cv::Mat& r0, const cv::Mat& t0, const cv::Mat& r1, const cv::Mat& t1, 
  		                     std::vector<cv::Point3d> pointcloud)
	{
		std::vector<double> res(pointcloud.size());

		std::vector<cv::Point3f> pt;
		for (int i = 0; i < pointcloud.size(); i++) {
			pt.push_back(Point3f(float(pointcloud[i].x), float(pointcloud[i].y), float(pointcloud[i].z)));
		}
		cv::Mat distortionCoefficients;
		std::vector<cv::Point2f> im0(pn0.size());
		std::vector<cv::Point2f> im1(pn1.size());

		cv::projectPoints(pt, r0, t0, K, distortionCoefficients, im0);
		cv::projectPoints(pt, r1, t1, K, distortionCoefficients, im1);

		//std::cout << "reprojection done" << std::endl;
		for (int i = 0; i < pointcloud.size(); i++) {
			res[i] = (  cv::norm(Point2f(im0[i].x-pn0[i].x, im0[i].y-pn0[i].y)) 
				      + cv::norm(Point2f(im1[i].x-pn1[i].x, im1[i].y-pn1[i].y)) ) / 2;
			//std::cout << pt[i] << "\t" << im0[i] << pn0[i] << "     \t" << im1[i] << pn1[i] << " " << res[i] << std::endl; 
		}

		return res;
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