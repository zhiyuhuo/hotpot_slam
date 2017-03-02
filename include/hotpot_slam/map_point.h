#ifndef MAP_POINT_H_
#define MAP_POINT_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common_headers.h"

class CMap;
class CRGBDFrame;
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


public: // functions
	CMapPoint()
	{
		_id = -1;
		_count_obv = 0;
		_p3d.x = 0;
		_p3d.y = 0;
		_p3d.z = 0;
		_dscp = Mat::zeros(1, 32, CV_8U);
	}

	CMapPoint(cv::Point3d p3d, cv::Mat dscp, int id = -1, int count_obv = 0)
	{
		_id = id;
		_count_obv = count_obv;
		_p3d = p3d;
		dscp.copyTo(_dscp);
	}

	~CMapPoint(){}

	void updateDscp(cv::Mat dscp)
	{
		_dscp = dscp.clone();
		_count_obv++;
	}

	void updateP3d(cv::Point3d p3d)
	{
		_p3d = p3d;
		_count_obv++;
	}

	void setPointID(int id)
	{
		_id = id;
	}

};

#endif