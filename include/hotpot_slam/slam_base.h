#ifndef SLAM_BASELINE_H_
#define SLAM_BASELINE_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common_headers.h"
#include "initializer.h"
#include "track_camera.h"

class CRGBDFrame;
class CKeyframe;
class CInitializer;
class CTrackResult;

class CSLAMBase
{
public: // data

	std::vector<CRGBDFrame> _rgbdframes;
	CRGBDFrame _rgbdframe_cur;
	CRGBDFrame _rgbdframe_lst;
	CRGBDFrame _init_rgbdframe;

public: // tool classes
	CParameterReader _para; // A stupid member I do not know why I write it

	// the instance that handle the initialization process. After running
	CInitializer _initializer; 
	// the instance that handel the local camera tracking process
	CTracker _tracker;

public: // data temperal and static
	// worldmap
	CMap _worldmap;
	// current using keyframe as the reference frame
	CKeyframe* _keyframe_now;
	vector<CKeyframe> _keyframes_list;

public: // State Machine Controller
	std::string _state;
	bool _iftracking;

public:
	CSLAMBase(){
		_tracker = CTracker(_para);
		_worldmap = CMap();
		_state = "init";
		_iftracking = true;
	}
	~CSLAMBase(){}

	int process()
	{
		if (_state == "init") {
			int r = initialize();
			if (r == 1)
			{
				_state = "running";
			}
		}
		
		else if (_state == "running") {
			trackCamera();
		}

	}

	int importDataToFrame(cv::Mat img_rgb, std::vector<float> vpoints)
	{
		CRGBDFrame rgbdframe(_para, img_rgb, vpoints);
		_rgbdframe_lst = _rgbdframe_cur;
		_rgbdframe_cur = rgbdframe;
		_rgbdframes.push_back(rgbdframe);
	}

	int initialize()
	{
		if (_rgbdframes.size() < 20) {
			// add enough frames for initialization. not in work now. just use the last frame capture.
			_initializer.addFrame(_rgbdframe_lst);	
			return 0;		
		}
		else {
			// initilization, firstly update world map.
			_initializer.initialize();
			_init_rgbdframe = _initializer._first_rgbdframe;
			// build the initliza frame
			_initializer.buildInitialMapFromInitialRGBDFrame(_worldmap);
			// make the keyframe pointer to the first keyframe in the world map
			_keyframe_now = _worldmap._keyframes[0];
			// show the initialize the world map
			_worldmap.showMapInfo();

			return 1;
		}
	}

	int trackCamera()
	{
		// track the keyframe now
		_tracker.trackKeyframe(_rgbdframe_cur, *(_keyframe_now));
		// update the tracking status for rviz
		_iftracking = _tracker._trackres._iftrack;
		// check is insert new key frame to the waiting list.
		if (checkIfAddNewKeyFrame()) {
			CKeyframe frame_new;
			frame_new.copyFrom(	_tracker._frame_new );
			_keyframes_list.push_back(frame_new);
		}

		std::cout << "the waiting list of the keyframes: " << std::endl;
		for (int i = 0; i < _keyframes_list.size(); i++) {
			std::cout << _keyframes_list[i]._Tvector << " " << _keyframes_list[i]._keypoints[0]._kp.pt << std::endl;
		}

		return 0;
	}

	bool checkIfAddNewKeyFrame() {
		return _tracker._trackres._ifshift;;
	}

	void trianglateTwoKeyFrames(CKeyframe& f0, CKeyframe& f1)
	{
		cv::Mat dscp0, dscp1;
		f0.buildDscpMat(dscp0);
		f1.buildDscpMat(dscp1);

		std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(NORM_HAMMING, true);
	    matcher.match(dscp1, dscp0, matches); // queryIdx, trainIdx

  		std::vector<cv::DMatch> good_matches;
  		int threshold = ORBMATCHTH;
		for( int i = 0; i < matches.size(); i++ ){
			if( matches[i].distance <= threshold) {
				good_matches.push_back(matches[i]);
			}
		}

		int id0, id1;
		std::vector<cv::DMatch> untrack_matches;
		for (int i = 0; i < good_matches.size(); i++) {
			id0 = good_matches[i].trainIdx;
			id1 = good_matches[i].queryIdx;
			if (!f0._keypoints[id0].ifTrack() && !f1._keypoints[id1].ifTrack()) {
				untrack_matches.push_back(good_matches[i]);
			}
		}

		// triangulate using official function
		cv::Mat R0, R1;
		eulerAnglesToRotationMatrix(f0._Rvector.at<double>(0, 0), f0._Rvector.at<double>(1, 0), f0._Rvector.at<double>(2, 0), R0);
		eulerAnglesToRotationMatrix(f1._Rvector.at<double>(0, 0), f1._Rvector.at<double>(1, 0), f1._Rvector.at<double>(2, 0), R1);
		cv::Mat T0 = f0._Tvector;
		cv::Mat T1 = f1._Tvector;
		std::vector<cv::Point2d> pn0(untrack_matches.size());
		std::vector<cv::Point2d> pn1(untrack_matches.size());
		for (int i = 0; i < untrack_matches.size(); i++)
		{
			id0 = untrack_matches[i].trainIdx;
			id1 = untrack_matches[i].queryIdx;
			pn0[i] = Point2d(double(f0._keypoints[id0]._kp.pt.x), double(f0._keypoints[id0]._kp.pt.y));
			pn1[i] = Point2d(double(f1._keypoints[id1]._kp.pt.x), double(f1._keypoints[id1]._kp.pt.y));
		}
		std::vector<cv::Point3d> p3d_untrack = triangulateTwoPixelSets(pn0, pn1, _tracker._Krgb, R0, T0, R1, T1);
		std::vector<double> err = computeReProjectionError(p3d, pn0, pn1, _tracker._Krgb, f0._Rvector, f0._Tvector, f1._Rvector, f1._Tvector);

		std::vector<cv::DMatch> valid_triangulate_matches;
		std::vector<cv::Point3d> p3d_totrack;
		for (int i = 0; i < untrack_matches[i].size(); i++)
		{
			if (cv::norm(pn0[i]-p1[i]) > 10 && err[i] < 1.0 && p3d_untrack[i].z > 0.5 && p3d_untrack[i].z < 4.0) {
				valid_triangulate_matches.push_back(untrack_matches[i]);
				p3d_totrack.push_back(p3d_untrack[i]);
			}
		}

		for (int i = 0; i < valid_triangulate_matches.size(); i++)
		{
			id0 = valid_triangulate_matches[i].trainIdx;
			id1 = valid_triangulate_matches[i].queryIdx;
			CMapPoint* mp = new CMapPoint(p3d_totrack[i], f1._keypoints[id1]._dscp);
			_worldmap.addMapPoint(mp);
			
		}		
	}

	cv::Point3d triagnlateOnePoint(const cv::Point2d& p0, const cv::Point2d& p1,const cv::Mat& K, 
		                           const cv::Mat& R0, const cv::Mat& T0, const cv::Mat& R1, const cv::Mat& T1)
	{
		cv::Point3d res(0, 0, 0);
		cv::Mat RT0(3,4,CV_64F);
		R0.copyTo(RT0(cv::Rect(0, 0, 3, 3)));
		T0.copyTo(RT0(cv::Rect(3, 0, 1, 3)));
		cv::Mat RT1(3,4,CV_64F);
		R1.copyTo(RT1(cv::Rect(0, 0, 3, 3)));
		T1.copyTo(RT1(cv::Rect(3, 0, 1, 3)));		

		std::vector<cv::Point2d> pn0(1, p0);
		std::vector<cv::Point2d> pn1(1, p1);
		cv::Mat pc4dmat(4,pn0.size(),CV_64FC4);

		triangulatePoints(K*RT0, K*RT1, pn0, pn1, pc4dmat);
		res.x = pc4dmat.at<double>(0,0) / pc4dmat.at<double>(3,0);
  		res.y = pc4dmat.at<double>(1,0) / pc4dmat.at<double>(3,0);
  		res.z = pc4dmat.at<double>(2,0) / pc4dmat.at<double>(3,0);

  		if (res.z < 0) {
  			res.x *= -1.0;
  			res.y *= -1.0;
  			res.z *= -1.0;
  		}
  			
  		return res;
	}

	/// triangulate points
  	std::vector<cv::Point3d> triangulateTwoPixelSets(const std::vector<cv::Point2d>& pn0, const std::vector<cv::Point2d>& pn1, 
  							 const cv::Mat& K, const cv::Mat& R0, const cv::Mat& T0, const cv::Mat& R1, const cv::Mat& T1)
	{
		std::vector<cv::Point3d> pointcloud(pn0.size());
  		// triangulate using official
		cv::Mat RT0(3,4,CV_64F);
		R0.copyTo(RT0(cv::Rect(0, 0, 3, 3)));
		T0.copyTo(RT0(cv::Rect(3, 0, 1, 3)));
		cv::Mat RT1(3,4,CV_64F);
		R1.copyTo(RT1(cv::Rect(0, 0, 3, 3)));
		T1.copyTo(RT1(cv::Rect(3, 0, 1, 3)));
		cv::Mat pc4dmat(4,pn0.size(),CV_64FC4);
  		cv::triangulatePoints(K*RT0, K*RT1, pn0, pn1, pc4dmat);
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

  		return pointcloud;
	}

	// compute the reprojection a 3d point cloud to two 2d images 
	std::vector<double> computeReProjectionError(std::vector<cv::Point3d> pointcloud, const std::vector<cv::Point2d>& pn0, const std::vector<cv::Point2d>& pn1, 
  							 const cv::Mat& K, const cv::Mat& r0, const cv::Mat& t0, const cv::Mat& r1, const cv::Mat& t1)
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

};



























#endif