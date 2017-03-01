#ifndef MAP_H_
#define MAP_H_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common_headers.h"
#include "map_point.h"

class CMapPoint;

// CMap is the only instance that manages all the points and frames.
class CMap
{
public: // members
	// total points that have been added to map. include the deleted points
	int _count_point;
	int _count_frame;
	// points
	std::map<int, CMapPoint> _worldpoints;
	// key frames
	std::map<int, CRGBDFrame> _keyframes;

public: // functions
	CMap()
	{
		_count_point = 0;
		_count_frame = 0;
		_worldpoints = std::map<int, CMapPoint>();
		_keyframes = std::map<int, CRGBDFrame>();
	}

	~CMap(){}

	void addMapPoint(CMapPoint point)
	{
		point.setPointID(_count_point);
		_worldpoints[_count_point++] = point;
	}

	void addKeyFrame(CRGBDFrame frame)
	{
		frame.setFrameID(_count_frame);
		_keyframes[_count_frame] = frame;
		_count_frame++;
	}

	void showMapInfo()
	{
		std::cout << "key frames number: " << _keyframes.size() << std::endl;
		for (int i = 0; i < _keyframes.size(); i++){
			std::cout << i << ": " << _keyframes[i]._id << std::endl;
		}
		std::cout << "world map points number: " << std::endl;
		for (int i = 0; i < _worldpoints.size(); i++) {
			std::cout << _worldpoints[i]._id << " " << _worldpoints[i]._p3d << " " << _worldpoints[i]._count_obv << std::endl;
			std::cout << _worldpoints[i]._kp_set[0].pt << _worldpoints[i]._dscp_set[0] << std::endl;
		}
	}
};

#endif