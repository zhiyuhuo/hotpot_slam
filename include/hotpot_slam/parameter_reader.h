#ifndef PARAMETER_READER_H_
#define PARAMETER_READER_H_

#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>

//#include "common_headers.h"

// TODO: write a read parameter function to read parameters from a file

using namespace std;

class CParameterReader
{
public:
	std::map<std::string, std::string> _data_directory;
	std::map<std::string, float> _camera_para;
	std::map<std::string, float> _orb_para;
	std::map<std::string, float> _pnp_para;
	std::map<std::string, float> _tracker_para;
	std::map<std::string, float> _mapper_para;
	std::map<std::string, float> _looper_para;

public:
	CParameterReader(std::string filename="/home/rokid/catkin_ws/src/hotpot_slam/parameters/parameters.txt") {
		int start_index = 1;
		int end_index  = -1;

		// data directory
		data_source = "/home/rokid/Documents/rgbd_dataset_freiburg1_room/";
		rgb_dir = "rgb/";
		rgb_extension = ".png";
		depth_dir = "depth/";
		depth_extension = ".png";

		// rgb camera parameters
		rgb_camera_width = 640;
		rgb_camera_height = 480;
		rgb_camera_cx = 303.9;
		rgb_camera_cy = 242.3;
		rgb_camera_fx = 544.4;
		rgb_camera_fy = 546.2;
		rgb_camera_d0 = 0.0;
		rgb_camera_d1 = 0.0;
		rgb_camera_d2 = 0.0;
		rgb_camera_d3 = 0.0;
		rgb_camera_d4 = 0.0;

		// depth camera parameters
		depth_camera_width = 640;
		depth_camera_height = 480;
		depth_camera_cx = 314.5;
		depth_camera_cy = 235.5;
		depth_camera_fx = 570.3;
		depth_camera_fy = 570.3;
		depth_camera_d0 = 0.0;
		depth_camera_d1 = 0.0;
		depth_camera_d2 = 0.0;
		depth_camera_d3 = 0.0;
		depth_camera_d4 = 0.0;
		depth_camera_scale = 1000.0;

		// ORB parameters
		orb_features = 1000;
		orb_scale = 1.2;
		orb_levels = 8;
		orb_iniThFAST = 20;
		orb_minThFAST = 8;
		knn_match_ratio = 0.8;

		// PnP parameters
		pnp_min_inliers = 10;
		pnp_min_matches = 15;

		// Tracker parameters
		tracker_max_lost_frame = 10;
		tracker_ref_frames = 5;

		// pose graph
		nearby_keyframes = 5;
		keyframe_min_translation = 0.25;
		keyframe_min_rotation = 0.25;
		loop_accumulate_error = 4.0;
		local_accumulate_error = 1.0;

		// Looper parameters
		looper_vocab_file = "/home/rokid/catkin_ws/src/hotpot_slam/vocab/ORBvoc.txt";
		looper_min_sim_score = 0.015;
		looper_min_interval = 60;

		// Mapper parameters
		mapper_resolution = 0.04;
		mapper_max_distance = 5.0;
	}

	~CParameterReader() {

	}

// temparal data
public:

	int start_index;
	int end_index;	

	// data directory
	std::string data_source;
	std::string rgb_dir;
	std::string rgb_extension;
	std::string depth_dir;
	std::string depth_extension;

	// rgb camera parameters
	int rgb_camera_width;
	int rgb_camera_height;
	float rgb_camera_cx;
	float rgb_camera_cy;
	float rgb_camera_fx;
	float rgb_camera_fy;
	float rgb_camera_d0;
	float rgb_camera_d1;
	float rgb_camera_d2;
	float rgb_camera_d3;
	float rgb_camera_d4;

	// depth camera parameters
	int depth_camera_width;
	int depth_camera_height;
	float depth_camera_cx;
	float depth_camera_cy;
	float depth_camera_fx;
	float depth_camera_fy;
	float depth_camera_d0;
	float depth_camera_d1;
	float depth_camera_d2;
	float depth_camera_d3;
	float depth_camera_d4;
	float depth_camera_scale;

	#// ORB parameters
	int orb_features;
	float orb_scale;
	int orb_levels;
	int orb_iniThFAST;
	int orb_minThFAST;
	int knn_match_ratio;

	// PnP parameters
	int pnp_min_inliers;
	int pnp_min_matches;

	// Tracker parameters
	int tracker_max_lost_frame;
	int tracker_ref_frames;

	// pose graph
	int nearby_keyframes;
	float keyframe_min_translation;
	float keyframe_min_rotation;
	float loop_accumulate_error;
	float local_accumulate_error;

	// Looper parameters
	std::string looper_vocab_file;
	float looper_min_sim_score;
	int looper_min_interval;

	// Mapper parameters
	float mapper_resolution;
	float mapper_max_distance;

};











#endif