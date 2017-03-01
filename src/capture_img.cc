#include <ros/ros.h>
#include <iostream>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "hotpot_slam/common_headers.h"

using namespace std;
using namespace cv;

const int IMGRGBWIDTH = 640;
const int IMGRGBHEIGHT = 480;
const int IMGRGBTYPE = CV_8UC3;

const int IMGDEPTHWIDTH = 640;
const int IMGDEPTHHEIGHT = 480;
const int IMGDEPTHTYPE = CV_16UC1;

Mat img_rgb;
Mat img_depth;
Mat img_show;

void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
void depthCallback(const sensor_msgs::ImageConstPtr& msg);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void CallBackFunc2(int event, int x, int y, int flags, void* userdata);

bool if_get_rgb;
bool if_get_depth;


int num;

int main(int argc, char **argv)
{
  num = 0;
  img_rgb = Mat::zeros(IMGRGBHEIGHT, IMGRGBWIDTH, IMGRGBTYPE);
  img_depth = Mat::zeros(IMGDEPTHHEIGHT, IMGDEPTHWIDTH, IMGDEPTHTYPE);

  if_get_rgb = false;
  if_get_depth = false;

  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;

  //ros::Subscriber sub_rgb = nh.subscribe("/camera/rgb/image_raw", 1, &rgbCallback);
  ros::Subscriber sub_depth = nh.subscribe("/camera/ir/image", 1, &depthCallback);

  // namedWindow("color", 1);
  // setMouseCallback("color", CallBackFunc, NULL);

  namedWindow("depth", 1);
  setMouseCallback("depth", CallBackFunc2, NULL);

  while (ros::ok())
  {

    if (
        //if_get_rgb 
        //&& 
        if_get_depth
        ) {
      //imshow("color", img_rgb);
      cv::normalize(img_depth, img_show, 0, 255, NORM_MINMAX, CV_32F);
      imshow("depth", img_show);
      waitKey(30);
      
    }
    
    ros::spinOnce();
  }

  return 0;
}

void rgbCallback(const sensor_msgs::ImageConstPtr& msg)
{
  if_get_rgb = true;
  cout << "rgb: " << msg->width << " " << msg->height << " " << msg->step << " " << msg->encoding << endl;
  memcpy(img_rgb.data, (uint8_t*)&msg->data[0], msg->step*msg->height);
}

void depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
  if_get_depth = true;
  cout << "depth: " << msg->width << " " << msg->height << " " << msg->step << " " << msg->encoding << endl;
  memcpy(img_depth.data, (float*)&msg->data[0], msg->step*msg->height);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == EVENT_LBUTTONDOWN )
     {
          cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
          string fname = "/home/rokid/Pictures/asus/rgb/" + to_string(num) + ".png";
          imwrite(fname, img_rgb);
          num++;
     }
}

void CallBackFunc2(int event, int x, int y, int flags, void* userdata)
{
     if  ( event == EVENT_LBUTTONDOWN )
     {
          cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
          string fname = "/home/rokid/Pictures/asus/depth/" + to_string(num) + ".png";
          imwrite(fname, img_show);
          num++;
     }
}