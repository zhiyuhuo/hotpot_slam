#ifndef INITIALIZER_H_
#define INITIALIZER_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common_headers.h"

class CMapPoint;
class CMap;

class CInitializer
{
public: // camera parameter and add-in frames
	CParameterReader _para;

	std::vector<CRGBDFrame> _frames;
	CRGBDFrame _frame_cur;
	CRGBDFrame _frame_lst;

public: // first key frame and first group of map points. The camera is at [I 0] for the first frame.
	CRGBDFrame _first_key_frame;

	std::vector<cv::KeyPoint> _keypoints;
	std::vector<cv::Point3d> _mappoints;
	std::vector<cv::Mat> _descriptors;

public:
	CInitializer(){
		
	}
	~CInitializer(){}

	/// add frame
	int addFrame(CRGBDFrame frame) {
		_frame_lst = _frame_cur;
		_frame_cur = frame;
		_frames.push_back(frame);
	}

	/// read data from raw rgb and vector format points
	int importDataToFrame(cv::Mat img_rgb, std::vector<float> vpoints)
	{
		CRGBDFrame frame(_para, img_rgb, vpoints);
		_frame_lst = _frame_cur;
		_frame_cur = frame;
		_frames.push_back(frame);
	}

	/// read data from rect raw rgb and rect raw depth
	int importDataToFrame(cv::Mat img_rgb, cv::Mat img_depth)
	{
		CRGBDFrame frame(_para, img_rgb, img_depth);
		_frame_lst = _frame_cur;
		_frame_cur = frame;
		_frames.push_back(frame);
	}

	// update map
	int updateWorldMap(CMap* worldmap)
	{
		int frameID = worldmap->_count_frame;
		worldmap->addKeyFrame(_first_key_frame);
		for (int i = 0; i < _first_key_frame._keypoints.size(); i++) {
			CMapPoint mpt = CMapPoint();
			Mat dscp = _first_key_frame._descriptor.row(i);
			mpt.addNewObservation(frameID, 
								  _first_key_frame._keypoints[i], 
								  _first_key_frame._pts_3d[i], 
								  dscp,
								  1.0);
			worldmap->addMapPoint(mpt);
		}
	}

	// start the initialization work after read enough data
	int initialize()
	{
		_first_key_frame = _frame_cur;

		initializeFromOneFramebyRGBPoints(_first_key_frame);

		std::cout << "result of init: " 
					<< _first_key_frame._keypoints.size() << " " 
					<< _first_key_frame._pts_3d.size() << " " 
					<< _first_key_frame._descriptor.size() << endl;

		return 0;
	}

	// for RGBD camera
	int initializeFromOneFramebyRGBPoints(CRGBDFrame& frame)
	{
		initializeFromOneFramebyRGBPoints(frame, frame._keypoints, frame._pts_3d, frame._descriptor);
		return 0;
	}

	// initialization using RGBD camera, return the key points, 3d map point positions and ORB descriptor of the kep points
	int initializeFromOneFramebyRGBPoints(CRGBDFrame frame, std::vector<KeyPoint>& keypoints,  std::vector<cv::Point3d>& mappoints, cv::Mat& descriptors)
	{
		// 0. cppy data
		cv::Mat img;
		std::vector<cv::Point3d> pts;
		for (int i = 0; i < frame._vpoints.size() / 3; i++) {
			pts.push_back(cv::Point3d(frame._vpoints[3*i], 
								  frame._vpoints[3*i+1], 
								  frame._vpoints[3*i+2]));
		}
		CParameterReader para = frame._parameter_reader;
		CCameraPara _rgbcam = CCameraPara( para.rgb_camera_width, para.rgb_camera_height,
  								para.rgb_camera_fx, para.rgb_camera_fy, para.rgb_camera_cx, para.rgb_camera_cy);
		CCameraPara _depthcam = CCameraPara(para.depth_camera_width, para.depth_camera_height,
								para.depth_camera_fx, para.depth_camera_fy, para.depth_camera_cx, para.depth_camera_cy);

		// 1. convert rgb image to gray image, smooth it
		cv::cvtColor(frame._img_rgb, img, CV_BGR2GRAY);
		//cv::medianBlur(img, img, 3);

		// 2. transfer points from depth camera space to rgb camera space. we assume there is no rotation
		cv::Point3d shift(0.025, 0, 0);
		for (int i = 0; i < pts.size(); i++)
			pts[i] = pts[i] - shift;

		// 3. compute the projection that the points to the rgb image, build a distance mask map
		cv::Mat mat_dist = cv::Mat::zeros(img.rows, img.cols, CV_32F);
		std::vector<cv::Point3d> xyz_in_rgb(img.rows * img.cols, cv::Point3d(-1, -1, -1));

		int u, v;
		for (int i = 0; i < pts.size(); i++) {
			if (pts[i].z != pts[i].z)
				continue;

			u = round(pts[i].x * _rgbcam._fx / pts[i].z + _rgbcam._cx);
			v = round(pts[i].y * _rgbcam._fy / pts[i].z + _rgbcam._cy);

			if (u < 0 || u >= img.cols || v < 0 || v >= img.rows) 
				continue;

			//std::cout << pts[i] << ", " << i%img.cols << " " << i/img.cols << ", " << u << " " << v << endl;

			if (mat_dist.at<float>(v, u) == 0) {
				mat_dist.at<float>(v, u) = pts[i].z;
				xyz_in_rgb[u + v*img.cols] = pts[i];
			}
			else if (mat_dist.at<float>(v, u) > 0) {
				if (pts[i].z < mat_dist.at<float>(v, u)) {
					mat_dist.at<float>(v, u) = pts[i].z;
				}
			}
		}

		// *********** draw the mast mat 
		cv::namedWindow("valid_mat");
		cv::imshow("valid_mat", mat_dist);
		cv::waitKey(0);
		cv::destroyWindow("valid_mat");
		// *********** draw mask end

		// 4. extract the key points from the gray image
		cv::medianBlur(img, img, 3);
		std::vector<KeyPoint> kps;
		cv::Mat dscp;
		cv::ORB orb(600, 1.2, 8, 31, 0, 2, ORB::FAST_SCORE, 31);
		orb.detect(img, kps);


		// 5. filter the key points not in good range and low corner score.
		// TO DO: search the distance not only in the correspond pixel but also on a small neighbour to smooth 
		std::cout << "kep points number: " << kps.size() << endl;
		std::vector<KeyPoint>::iterator it;
		for (it = kps.begin(); it != kps.end(); it++) {
			int idx = round(it->pt.x) + round(it->pt.y) * img.cols;
			if (mat_dist.at<float>(round(it->pt.y), round(it->pt.x)) <= 0.5 
				|| mat_dist.at<float>(round(it->pt.y), round(it->pt.x)) > 5.0
				|| xyz_in_rgb[idx].z <= 0) {
				kps.erase(it);
				it--;
			}
		}
		std::cout << "kep points number after filter: " << kps.size() << endl;

		// 6. draw points in rviz in the main function (drawing is not in this class)
	    // fetch xyz for the left key points
	    std::vector<cv::Point3d> xyz_kps;
	    for (int i = 0; i < kps.size(); i++) {
	    	int idx = round(kps[i].pt.x) + round(kps[i].pt.y) * img.cols;
	    	//std::cout << "kp: " << i << " " << kps[i].pt << " " << idx << " " << xyz_in_rgb[idx] << endl;
	    	xyz_kps.push_back(xyz_in_rgb[idx]);

	    }

	    // ********* draw key points in gray image after filter 
		Mat img_show2;
	    cv::drawKeypoints( img, kps, img_show2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT ); // DRAW_RICH_KEYPOINTS
	    cv::namedWindow("init_keypoints");
	    cv::imshow( "init_keypoints", img_show2 );
	    cv::waitKey(0); 
	    cv::destroyWindow("init_keypoints");
	    img_show2.release();

	    // ********* draw end

	    keypoints = kps;
	    mappoints = xyz_kps;
	    orb.compute(img, kps, dscp);
	    dscp.copyTo(descriptors);
		return 0;
	}

	// for Mono camera TO DO
};

#endif