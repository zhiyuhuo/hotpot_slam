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
	bool _iftracking;

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

	int initialize()
	{
		if (_frames.size() < 20) {
			_initializer.addFrame(_frame_lst);	
			return 0;		
		}
		else {
			// initilization, firstly update world map.
			_initializer.initialize();
			_initializer.buildInitialMapPointsFromInitialKeyFrame(_worldmap);
			_first_key_frame = _initializer._first_key_frame;
			_worldmap.showMapInfo();
			_tracker.setReferenceFrame(_first_key_frame);
			return 1;
		}
	}

	int trackCamera()
	{
		_iftracking = _tracker.trackIter(_frame_cur);
		if (checkIfAddNewKeyFrame())
		{
			_worldmap.addKeyFrame(_tracker._frames.back());
		}
		_tracker._frames.clear();
	}

	bool checkIfAddNewKeyFrame() {
		// 1. grab the new income frame from _tracker
		bool res = false;
		if ( _tracker._frames.size() > 0 ) {
			CRGBDFrame f = _tracker._frames.back();
			if ( _worldmap._keyframes.size() <= 0 ) {
				res = true;
			}
			else {
				double max_r = 0.0;
				int nearest_frame_id = 0;
				std::map<int, CRGBDFrame>::iterator it;
				for (it = _worldmap._keyframes.begin(); it != _worldmap._keyframes.end(); it++) {
					double r = scoreMatchOfTwoFrames(f, it->second);
					std::cout << it->first << ": " << r << std::endl;
					if (r > max_r) {
						max_r = r;
						nearest_frame_id = it->first;
					}
				}

				_tracker.setReferenceFrame(_worldmap._keyframes[nearest_frame_id]);

				if (max_r < 0.5) {
					res = true;
				}
			}
		}

		return res;
	}

	double scoreMatchOfTwoFrames(CRGBDFrame f0, CRGBDFrame f1) {

		std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(NORM_HAMMING, true);
	    matcher.match(f0._dscp, f1._dscp, matches);

		// 4. filter low score match, only keep the high score matches
		//	  we will have several filters. 1st from score, 2nd from fake optical flow
		//    Here we will compute a fake optical flow and abort all the points which does not match the flow
  		std::vector<cv::DMatch> good_matches;
  		int threshold = 64;
		for( int i = 0; i < matches.size(); i++ ){
			if( matches[i].distance <= threshold) {
				good_matches.push_back(matches[i]);
			}
		}

		double res = double(good_matches.size()) / double(f0._kps.size()+f1._kps.size()-good_matches.size());
		return res;
	}

};



























#endif