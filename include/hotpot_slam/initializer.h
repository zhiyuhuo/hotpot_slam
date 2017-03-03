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
class CRGBDFrame;
class CKeyframe;

class CInitializer
{
public: // camera parameter and add-in frames
	CParameterReader _para;

	std::vector<CRGBDFrame> _frames;
	CRGBDFrame _frame_cur;
	CRGBDFrame _frame_lst;

public: // first key frame and first group of map points. The camera is at [I 0] for the first frame.
	CRGBDFrame _first_rgbd_frame;
	CKeyframe _init_keyframe;

	std::vector<cv::KeyPoint> _kps;
	std::vector<cv::Point3d> _p3d;
	std::vector<cv::Mat> _dscp;

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

	// start the initialization work after read enough data
	int initialize()
	{
		_first_rgbd_frame = _frame_cur;

		initializeFromOneFramebyRGBPoints(_first_rgbd_frame);

		std::cout << "result of init: " 
					<< _first_rgbd_frame._kps.size() << " " 
					<< _first_rgbd_frame._p3d.size() << " " 
					<< _first_rgbd_frame._dscp.size() << endl;

		return 0;
	}

	// for RGBD camera
	int initializeFromOneFramebyRGBPoints(CRGBDFrame& frame)
	{
		initializeFromOneFramebyRGBPoints(frame, frame._kps, frame._p3d, frame._dscp);
		return 0;
	}

	int initializeFromTenNeighborFrames(vector<CRGBDFrame> frames) {


		for (int i = 0; i < frames.size(); i++) {

		}

		return 0;
	}

	// initialization using RGBD camera, return the key points, 3d map point positions and ORB descriptor of the kep points
	int initializeFromOneFramebyRGBPoints(CRGBDFrame frame, std::vector<KeyPoint>& keypoints,  std::vector<cv::Point3d>& mappoints, cv::Mat& descriptors)
	{
		// 0. cppy data
		cv::Mat img;
		std::vector<cv::Point3d> p3d;
		for (int i = 0; i < frame._vpoints.size() / 3; i++) {
			p3d.push_back(cv::Point3d(frame._vpoints[3*i], 
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
		for (int i = 0; i < p3d.size(); i++)
			p3d[i] = p3d[i] - shift;

		// 3. compute the projection that the points to the rgb image, build a distance mask map
		cv::Mat mat_dist = cv::Mat::zeros(img.rows, img.cols, CV_32F);

		int u, v;
		std::vector<cv::Point3d> p3d_rgb(img.cols * img.rows);
		for (int i = 0; i < p3d.size(); i++) {
			if (p3d[i].z != p3d[i].z)
				continue;

			u = round(p3d[i].x * _rgbcam._fx / p3d[i].z + _rgbcam._cx);
			v = round(p3d[i].y * _rgbcam._fy / p3d[i].z + _rgbcam._cy);

			if (u < 0 || u >= img.cols || v < 0 || v >= img.rows) 
				continue;

			int idx = u + v * img.cols;
			//std::cout << p3d[i] << ", " << i%img.cols << " " << i/img.cols << ", " << u << " " << v << endl;

			if (mat_dist.at<float>(v, u) == 0) {
				mat_dist.at<float>(v, u) = p3d[i].z;
				p3d_rgb[idx] = p3d[i];
			}
			else if (mat_dist.at<float>(v, u) > 0) {
				if (p3d[i].z < mat_dist.at<float>(v, u)) {
					mat_dist.at<float>(v, u) = p3d[i].z;
					p3d_rgb[idx] = p3d[i];
				}
			}
		}

		// *********** draw the mast mat 
		cv::namedWindow("valid_region_mat");
		cv::imshow("valid_region_mat", mat_dist);
		cv::waitKey(0);
		cv::destroyWindow("valid_mat");
		// *********** draw mask end

		// 4. extract the key points from the gray image
		cv::medianBlur(img, img, 3);
		std::vector<KeyPoint> kps;
		cv::Mat dscp;
		cv::ORB orb(1000, 1.2, 8, 31, 0, 2, ORB::FAST_SCORE, 31);
		orb.detect(img, kps);


		// 5. filter the key points not in good range and low corner score.
		// TO DO: search the distance not only in the correspond pixel but also on a small neighbour to smooth 
		std::cout << "kep points number: " << kps.size() << endl;
		selectValidPoints(mat_dist, kps);
		std::cout << "kep points number after filter: " << kps.size() << endl;

		// 6. draw points in rviz in the main function (drawing is not in this class)
	    // fetch xyz for the left key points
	    std::vector<cv::Point3d> p3d_kps;
	    for (int i = 0; i < kps.size(); i++) {
	    	int idx = round(kps[i].pt.x) + round(kps[i].pt.y) * img.cols;
	    	//std::cout << "kp: " << i << " " << kps[i].pt << " " << idx << " " << xyz_in_rgb[idx] << endl;
	    	p3d_kps.push_back(p3d_rgb[idx]);

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
	    mappoints = p3d_kps;
	    orb.compute(img, kps, dscp);
	    dscp.copyTo(descriptors);
		return 0;
	}

	int buildInitialMapFromInitialRGBDFrame(CMap& worldmap)
	{
		CKeyframe* keyframe = new CKeyframe();
		for (int i = 0; i < _first_rgbd_frame._kps.size(); i++) {
			CMapPoint* pt = new CMapPoint(_first_rgbd_frame._p3d[i], _first_rgbd_frame._dscp.row(i));
			worldmap.addMapPoint(pt);

			CKeypoint kp(_first_rgbd_frame._kps[i], _first_rgbd_frame._dscp.row(i));
			kp.trackToMapPoint(pt);
			keyframe->addKeypoint(kp);
		}
		worldmap.addKeyframe(keyframe);
	}

 	// this function will select valid 3d points by the distance and neigbour (not in edge)
	int selectValidPoints(cv::Mat mat_dist, std::vector<cv::KeyPoint>& kps)
	{
		std::vector<KeyPoint>::iterator it;
		int u, v;
		for (it = kps.begin(); it != kps.end(); it++) {
			u = round(it->pt.x);
			v = round(it->pt.y);
			if (mat_dist.at<float>(v, u) <= 0.5	|| mat_dist.at<float>(v, u) > 4.0) {
				kps.erase(it);
				it--;
			}
			else if (getMaxDistanceFromNeighourPixel(mat_dist, u, v, 5) > 0.01){
				kps.erase(it);
				it--;
			}

		}	
		return kps.size();	
	}

	// this function return the max distance that the neigbor pixels to the key point.u and v are choosen here, L is an odd number
	float getMaxDistanceFromNeighourPixel(const cv::Mat& mat_dist, int u, int v, int L) {
		float dist = 0;
		float dp = mat_dist.at<float>(v, u);
		float d;
		for (int y = v-L/2; y <= v+L/2; y++ ) {
			for (int x = u-L/2; x <= u+L/2; x++ ) {
				if (x >=0 && x < mat_dist.cols && y >=0 && y < mat_dist.rows) {
					d = mat_dist.at<float>(y, x);
					if (abs(d-dp) > dist) {
						dist = abs(d-dp);
					}
				}
			}
		}

		return dist;
	}

	// for Mono camera TO DO
};

#endif