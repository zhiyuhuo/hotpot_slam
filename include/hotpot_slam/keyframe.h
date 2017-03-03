#ifndef FRAME_H_
#define FRAME_H_

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
#include "keypoint.h"

using namespace std;
using namespace cv;

class CMap;
class CMapPoint;
class CKeypoint;
class CKeyframe
{
public:
	/// unique frame ID when it is inserted to map
	int _id;

	/// camera pose of the key frame
	cv::Mat _Rvector;
	cv::Mat _Tvector;

	/// CKeypoint members
	vector<CKeypoint> _keypoints;

	/// bobw data. TODO. this is for relocalization and global optimization. 
	//DBoW2::BowVector bowVec;

public:
	CKeyframe()
	{
		_id = -1;
		_Rvector = cv::Mat::zeros(3, 1, CV_64F);
		_Tvector = cv::Mat::zeros(3, 1, CV_64F);
	}

	~CKeyframe()
	{

	}

	void reset()
	{
		_Rvector.at<double>(0, 0) = 0;
		_Rvector.at<double>(1, 0) = 0;
		_Rvector.at<double>(2, 0) = 0;
		_Tvector.at<double>(0, 0) = 0;
		_Tvector.at<double>(1, 0) = 0;
		_Tvector.at<double>(2, 0) = 0;
		_id = -1;
		_keypoints.clear();
	}

	void setFrameID(int id) 
	{
		_id = id;
	}

	void addKeypoint(CKeypoint kp)
	{
		_keypoints.push_back(kp);
	}

	void setRT(cv::Mat rvector, cv::Mat tvector)
	{
		rvector.copyTo(_Rvector);
		tvector.copyTo(_Tvector);
	}

	void copyFrom(CKeyframe& kf)
	{
		_id = kf._id;
		kf._Rvector.copyTo(_Rvector);
		kf._Tvector.copyTo(_Tvector);
		_keypoints = kf._keypoints;
	}

	void buildDscpMat(cv::Mat& mat)
	{
		cv::Mat dscp = cv::Mat::zeros(_keypoints.size(), ORBDSCP_L, CV_8U);
		for (int i = 0; i < _keypoints.size(); i++) {
			_keypoints[i]._dscp.copyTo(dscp.row(i));
		}
		dscp.copyTo(mat);
		dscp.release();
	}

};







#endif