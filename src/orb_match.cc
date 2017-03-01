#include <ros/ros.h>
#include <iostream>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/Marker.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>

#include "hotpot_slam/common_headers.h"

using namespace std;
using namespace cv;

const int IMGRGBWIDTH = 640;
const int IMGRGBHEIGHT = 480;
const int IMGRGBTYPE = CV_8UC3;

Mat img_rgb;
Mat first_rgb;
Mat img_rgb_lst;

void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
void imageMatch(Mat img_lst, Mat img_cur);

bool if_get_rgb;

int frame_count;

int main(int argc, char **argv)
{
  // initialize data container
  img_rgb = cv::Mat::zeros(IMGRGBHEIGHT, IMGRGBWIDTH, IMGRGBTYPE);
  first_rgb = cv::Mat::zeros(IMGRGBHEIGHT, IMGRGBWIDTH, IMGRGBTYPE);

  if_get_rgb = false;

  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;

  ros::Subscriber sub_rgb = nh.subscribe("/camera/rgb/image_rect_color", 1, &rgbCallback);

  frame_count = 0;
  int first_count = 30;

  while (ros::ok())
  {
    if ( if_get_rgb ) {

      if (frame_count == first_count) {
        first_rgb = img_rgb.clone();
      }
      else if (frame_count > first_count) {
        imageMatch(img_rgb_lst, img_rgb);
      }

      if ( frame_count % 10 == 0) {
        img_rgb.copyTo(img_rgb_lst);
        cout << first_count << "new reference frame" << endl;
      }
      cv::imshow("rgb", img_rgb);
      cv::waitKey(1);
      frame_count++;
    }

    ros::spinOnce();
  }
  cv::destroyWindow("rgb");

  return 0;
}

void rgbCallback(const sensor_msgs::ImageConstPtr& msg)
{
  if_get_rgb = true;
  std::memcpy(img_rgb.data, (uint8_t*)&msg->data[0], msg->step*msg->height);
  cv::cvtColor(img_rgb, img_rgb, CV_BGR2RGB);
}

void imageMatch(Mat img_color_lst, Mat img_color_cur)
{
    cv::Mat img_gray_lst;
    cv::cvtColor(img_color_lst, img_gray_lst, CV_BGR2GRAY);
    cv::Mat img_lst = img_gray_lst;

    cv::Mat img_gray_cur;
    cv::cvtColor(img_color_cur, img_gray_cur, CV_BGR2GRAY);
    cv::Mat img_cur = img_gray_cur;

    // 2. detect key points on the current image, extract the ORB features and compute descriptor for the current image
    cv::ORB orb(600, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
    std::vector<cv::KeyPoint> kps_lst, kps_cur;
    cv::Mat dscp_lst, dscp_cur;
    orb.detect(img_lst, kps_lst);
    orb.detect(img_cur, kps_cur);
    orb.compute(img_lst, kps_lst, dscp_lst);
    orb.compute(img_cur, kps_cur, dscp_cur);

    // 3. match two frames, get the matching 
    std::vector<cv::DMatch> matches;
    cv::BFMatcher matcher(NORM_HAMMING, true);
    matcher.match(dscp_cur, dscp_lst, matches);

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < matches.size(); i++)
    {
      if (matches[i].distance < 32) {
        good_matches.push_back(matches[i]);
      }
    }

    Mat img_match;
    drawMatches(img_gray_cur, kps_cur, img_color_lst, kps_lst, good_matches, img_match);

    cout << good_matches.size() << endl;

    imshow("matches", img_match);
    waitKey(1);

}

