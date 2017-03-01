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
class CInitializer;
class CSLAMBase
{
public: // data

	std::vector<CRGBDFrame> _frames;
	CRGBDFrame _frame_cur;
	CRGBDFrame _frame_lst;
	CRGBDFrame _first_key_frame;

public: // tool classes
	CParameterReader _para; // A stupid member I do not know why I write it

	// the instance that handle the initialization process. After running
	CInitializer _initializer; 
	// the instance that handel the local camera tracking process
	CTracker _tracker;
	// worldmap
	CMap _worldmap;

public: // data container


public: // State Machine Controller
	std::string _state;

public:
	CSLAMBase(){
		_tracker = CTracker(_para);
		_worldmap = CMap();
		_state = "init";
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
		CRGBDFrame frame(_para, img_rgb, vpoints);
		_frame_lst = _frame_cur;
		_frame_cur = frame;
		_frames.push_back(frame);
	}

	int importDataToFrame(cv::Mat img_rgb, cv::Mat img_depth)
	{
		CRGBDFrame frame(_para, img_rgb, img_depth);
		_frame_lst = _frame_cur;
		_frame_cur = frame;
		_frames.push_back(frame);
	}

	int initialize()
	{
		if (_frames.size() < 20) {
			_initializer.addFrame(_frame_lst);	
			return 0;		
		}
		else {
			// initilization, firstly update world map.
			_initializer.initialize();
			_first_key_frame = _initializer._first_key_frame;
			_initializer.updateWorldMap(&_worldmap);
			_worldmap.showMapInfo();

			_tracker.addKeyFrame(_first_key_frame);
			return 1;
		}
	}

	int trackCamera()
	{
		_tracker.trackIter(_frame_cur);
	}

};



























#endif