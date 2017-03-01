#ifndef CAMERA_PARA_H_
#define CAMERA_PARA_H_

#include <stdlib.h>
#include "common_headers.h"
#include "parameter_reader.h"

class CCameraPara
{
public:

	CCameraPara()
	{
		
	}

	CCameraPara(int w, int h, float fx, float fy, float cx, float cy, 
				float d0=0, float d1=0, float d2=0, float d3=0, float d4=0) 
	{
		_width = w;
		_height = h;
		
		_fx = fx;
		_fy = fy;
		_cx = cx;
		_cy = cy;
		_d0 = d0;
		_d1 = d1;
		_d2 = d2;
		_d3 = d3;
		_d4 = d4;
	}

	~CCameraPara()
	{}

public:
	float _fx;
	float _fy;
	float _cx;
	float _cy;
	float _d0;
	float _d1;
	float _d2;
	float _d3;
	float _d4;

	int _width;
	int _height;

/*	void readParaFromReader(CParameterReader reader){

	}*/
};


#endif