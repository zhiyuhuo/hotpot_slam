#ifndef MAP_POINT_H_
#define MAP_POINT_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common_headers.h"

class CMapPoint
{
public: // members

	// point id
	int _id;
	// the number being observed
	int _count_obv;
	// 3D point position in the world coordinate (origin at the camera pose of the first frame)
	cv::Point3d _p3d;	
	// the average ORB descriptor of the map point (however, I hope it can be the one with the highest probabiliry)
	cv::Mat _dscp;
	// the pixel position of the map point in each frame it observed.
	std::map<int, cv::KeyPoint> _kp_set; // Not in Use 
	// the descriptor of the point in each frame where the point is detected.
	std::map<int, cv::Mat> _dscp_set; // Not in Use
	// the quality, or we can say the probability of the point in each frame (actually, I do not know how to evaluate it |_|)
	std::map<int, double> _qlt_list; // Not in Use

public: // functions
	CMapPoint()
	{
		_id = -1;
		_count_obv = 0;
		_p3d.x = 0;
		_p3d.y = 0;
		_p3d.z = 0;
		_dscp = Mat::zeros(1, 32, CV_8U);
		_kp_set = std::map<int, cv::KeyPoint>();
		_dscp_set = std::map<int, cv::Mat>();
		_qlt_list = std::map<int, double>();
	}
	~CMapPoint(){}

	void addNewObservation(int frmid, cv::KeyPoint kp, cv::Point3d p3d, cv::Mat dscp, double quality)
	{
		_p3d = p3d;
		_kp_set[frmid] = kp;
		_dscp_set[frmid] = dscp.clone();
		_dscp = dscp.clone();
		_qlt_list[frmid] = quality;
		_count_obv++;
	}

	void setPointID(int id)
	{
		_id = id;
	}

};

#endif