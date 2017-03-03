#ifndef RGBD_FRAME_H_
#define RGBD_FRAME_H_

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <map>
#include <pcl/point_types.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

#include "common_headers.h"
#include "parameter_reader.h"

using namespace std;
using namespace cv;

class CMap;
class CMapPoint;
class CRGBDFrame
{
public:
	/// unique frame ID when it is inserted to map
	int _id;

	/// read parameters;
	CParameterReader _parameter_reader;

	/// input data
	cv::Mat _img_rgb;
	std::vector<float> _vpoints;

	/// frame feature data
	/// Keypoints + 3D position (this will be obtained using triangulate, only calculated if the frame become a KEY frame)
	/// + ORB feature (these data will be filled if the frame does not is to be the key frame)
	std::vector<cv::KeyPoint> _kps;
	std::vector<cv::Point3d> _p3d;
	cv::Mat _dscp;

	/// camera pose 
	cv::Mat _Rvector;
	cv::Mat _Tvector;

public:
	CRGBDFrame()
	{
		_Rvector = cv::Mat::zeros(3, 1, CV_64F);
		_Tvector = cv::Mat::zeros(3, 1, CV_64F);
	}

	CRGBDFrame(CParameterReader parameter_reader, cv::Mat img_rgb, std::vector<float> vpoints)
	{
		_parameter_reader = parameter_reader;
		_img_rgb = img_rgb.clone();
		_vpoints = vpoints;
	}

	CRGBDFrame(CParameterReader parameter_reader, cv::Mat img_rgb, std::vector<cv::KeyPoint> kps, std::vector<cv::Point3d> p3d, cv::Mat dscp) {
		img_rgb.copyTo(_img_rgb);
		_kps = kps;
		_p3d = p3d;
		dscp.copyTo(_dscp);
	}

	~CRGBDFrame()
	{
		_img_rgb.release();
	}

	void setFrameID(int id) 
	{
		_id = id;
	}

	void updateFrameFeatureData(std::vector<cv::KeyPoint> kps, cv::Mat dscp)
	{
		_kps = kps;
		_dscp = dscp.clone();
	}

	void updateFrameFeatureData(cv::Mat img_rgb, std::vector<cv::KeyPoint> kps, std::vector<cv::Point3d> p3d, cv::Mat dscp)
	{
		img_rgb.copyTo(_img_rgb);
		_kps = kps;
		_p3d = p3d;
		_dscp = dscp.clone();
	}

};







#endif