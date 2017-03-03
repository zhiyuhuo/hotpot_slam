#ifndef KEYPOINT_MY_H_
#define KEYPOINT_MY_H_

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
class CKeypoint
{
public:

	/// input data
	cv::KeyPoint _kp;
	cv::Mat _dscp;

	/// the pointer to a map point if the kp is tracked;
	bool _ifTracked;
	CMapPoint *_pmappoint;

public:
	CKeypoint()
	{
		_dscp = cv::Mat::zeros(1, ORBMATCHTH, CV_8U);
		_ifTracked = false;
		_pmappoint = NULL;
	}

	CKeypoint(cv::KeyPoint kp, cv::Mat dscp)
	{
		_kp = kp;
		dscp.copyTo(_dscp);
		_ifTracked = false;
		_pmappoint = NULL;
	}

	~CKeypoint()
	{
		_dscp.release();
	}

	void setKeyPoint(cv::KeyPoint kp) 
	{
		_kp = kp;
	}

	void setDescriptor(cv::Mat dscp) 
	{
		dscp.copyTo(_dscp);
	}

	void trackToMapPoint(CMapPoint *pmappoint)
	{
		_pmappoint = pmappoint;
		_ifTracked = true;
	}

	bool ifTrack()
	{
		if (_pmappoint == NULL)
			return false;
		return _ifTracked;
	}

};







#endif