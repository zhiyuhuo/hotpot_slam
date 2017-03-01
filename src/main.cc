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

const int IMGDEPTHWIDTH = 640;
const int IMGDEPTHHEIGHT = 480;
const int IMGDEPTHTYPE = CV_32FC1;

Mat img_rgb;
Mat img_depth;
Mat img_color;
std::vector<float> vpoints;

void rgbCallback(const sensor_msgs::ImageConstPtr& msg);
void depthCallback(const sensor_msgs::ImageConstPtr& msg);
void pointsCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
void broadcastTF();
void editPointCloudMsg(std::vector<Point3d> pts, sensor_msgs::PointCloud& msg);
void editMarkerMsg(cv::Mat r,cv::Mat t, visualization_msgs::Marker& marker);
std::vector<double> rotationMatrixToQuanterion(Mat r);
std::vector<double> eulerToQuaternion(double pitch, double roll, double yaw);

bool if_get_rgb;
bool if_get_depth;
bool if_get_points;

uint32_t rgb_time_ns;
uint32_t depth_time_ns;

ros::Publisher pub_pc;
int id_pc;
int frame_count;

int main(int argc, char **argv)
{
  rgb_time_ns = 0;
  depth_time_ns = 0;
  id_pc = 0;

  // initialize data container
  img_rgb =cv::Mat::zeros(IMGRGBHEIGHT, IMGRGBWIDTH, IMGRGBTYPE);
  img_depth =cv::Mat::zeros(IMGDEPTHHEIGHT, IMGDEPTHWIDTH, IMGDEPTHTYPE);
  img_color =cv::Mat::zeros(IMGDEPTHHEIGHT, IMGDEPTHWIDTH, IMGRGBTYPE);

  vpoints = std::vector<float>(IMGDEPTHHEIGHT*IMGDEPTHWIDTH*3, 0);

  if_get_rgb = false;
  if_get_depth = false;
  if_get_points = false;

  ros::init(argc, argv, "image_listener");
  ros::NodeHandle nh;

  ros::Subscriber sub_rgb = nh.subscribe("/camera/rgb/image_rect_color", 1, &rgbCallback);
  //ros::Subscriber sub_depth = nh.subscribe("/camera/depth/image_rect", 1, &depthCallback);
  ros::Subscriber sub_points = nh.subscribe("/camera/depth/points", 1, &pointsCallback);

  pub_pc = nh.advertise<sensor_msgs::PointCloud> ("/hotpot_slam/points", 1);

  sensor_msgs::PointCloud msg_pointcloud;

  ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker", 1);

  visualization_msgs::Marker msg_marker;

  frame_count = 0;

  CSLAMBase slam;

  CParameterReader para;
  CTracker track(para);
  CRGBDFrame frame_reference;
  ros::Rate rate(30);
  cv::Mat Rvector =cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat Tvector =cv::Mat::zeros(3, 1, CV_64F);
  cv::namedWindow("color");
  cv::moveWindow("color", 1600, 300);
  while (ros::ok())
  {
    if ( if_get_rgb && if_get_points) {

      slam.importDataToFrame(img_rgb, vpoints);
      slam.process();

      cv::imshow("color", img_rgb);
      cv::waitKey(1);

      slam._tracker._Rvector.copyTo(Rvector);
      slam._tracker._Tvector.copyTo(Tvector);

      // publish camera pose
      editMarkerMsg(Rvector, Tvector, msg_marker);
      while (marker_pub.getNumSubscribers() < 1)
      {
        if (!ros::ok())
        {
          return 0;
        }
        ROS_WARN_ONCE("Please create a subscriber to the marker");
        sleep(1);
      }
      marker_pub.publish(msg_marker);

      // publish worldmap points
      broadcastTF();
      if (slam._tracker._keyframes.size() <= 0)
        continue;
      editPointCloudMsg(slam._tracker._keyframes.back()._pts_3d, msg_pointcloud);
      pub_pc.publish(msg_pointcloud);
      id_pc++;

    }

    ros::spinOnce();
    rate.sleep();
  }
  cv::destroyWindow("color");

  return 0;
}

void rgbCallback(const sensor_msgs::ImageConstPtr& msg)
{
  if_get_rgb = true;
  rgb_time_ns = msg->header.stamp.nsec;
  //rgb_time_ns = msg->header.stamp.to_sec();
  //std::cout << "rgb: " << msg->width << " " << msg->height << " " << msg->step << " " << msg->encoding << endl;
  std::memcpy(img_rgb.data, (uint8_t*)&msg->data[0], msg->step*msg->height);
  cv::cvtColor(img_rgb, img_rgb, CV_BGR2RGB);
}

void depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
  if_get_depth = true;
  depth_time_ns = msg->header.stamp.nsec;
  //depth_time_ns = msg->header.stamp.to_sec();
  //std::cout << "depth: " << msg->width << " " << msg->height << " " << msg->step << " " << msg->encoding << endl;
  std::memcpy(img_depth.data, (float*)&msg->data[0], msg->step*msg->height);
}

void pointsCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
  if_get_points = true;
  //std::cout << "points: " << msg->width << " " << msg->height << " " << msg->point_step << " " << msg->row_step << endl;
  int step = (int)msg->point_step;
  float x,y,z;
  for (int i = 0; i < msg->width * msg->height; i++) {

    std::memcpy(&x, &msg->data[step*i], 4);
    std::memcpy(&y, &msg->data[step*i+4], 4);
    std::memcpy(&z, &msg->data[step*i+8], 4);

    vpoints[3*i] = x;
    vpoints[3*i+1] = y;
    vpoints[3*i+2] = z;
  }
}

void broadcastTF()
{
  static tf::TransformBroadcaster br;
  tf::Transform transform;
  transform.setOrigin( tf::Vector3(0.0, 0.0, 0.0) );
  transform.setRotation( tf::Quaternion(0, 0, 0) );
  br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "points_frame")); //don't know what to pass through
}

void editPointCloudMsg(std::vector<Point3d> pts, sensor_msgs::PointCloud& msg)
{
  msg.header.seq = id_pc;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "points_frame";

  msg.channels.resize(1);
  msg.channels[0].name = "intensities";
  msg.channels[0].values.resize(pts.size());

  msg.points.resize(pts.size());
  //std::cout << "Point Cloud: " << endl;
  for (int i = 0; i < msg.points.size(); i++) {
    msg.points[i].x = pts[i].x;
    msg.points[i].y = pts[i].y;
    msg.points[i].z = pts[i].z;
    msg.channels[0].values[i] = 255;
    //std::cout << msg.points[i].x << " " << msg.points[i].y << " " << msg.points[i].z << endl;
  }

}

void editMarkerMsg(cv::Mat r,cv::Mat t, visualization_msgs::Marker& marker)
{
  // Set the frame ID and timestamp.  See the TF tutorials for information on these.
    marker.header.frame_id = "/points_frame";
    marker.header.stamp = ros::Time::now();

    // Set the namespace and id for this marker.  This serves to create a unique ID
    // Any marker sent with the same namespace and id will overwrite the old one
    marker.ns = "basic_shapes";
    marker.id = 0;

    // Set the marker type.  CUBE
    marker.type = visualization_msgs::Marker::ARROW;

    // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    marker.action = visualization_msgs::Marker::ADD;

    // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
    std::vector<double> q = eulerToQuaternion(-r.at<double>(0,0)+ 1.5708, -r.at<double>(1,0), -r.at<double>(2,0));
    marker.pose.position.x =  -t.at<double>(0,0);
    marker.pose.position.y =  -t.at<double>(1,0);
    marker.pose.position.z =  -t.at<double>(2,0);
    marker.pose.orientation.x = q[0];
    marker.pose.orientation.y = q[1];
    marker.pose.orientation.z = q[2];
    marker.pose.orientation.w = q[3];

    //std::cout << "marker pose:" << endl << marker.pose << endl;
    //std::cout << "marker pose:" << endl << marker.pose.position << endl;

    // Set the scale of the marker -- 1x1x1 here means 1m on a side
    marker.scale.x = 0.10;
    marker.scale.y = 0.01;
    marker.scale.z = 0.01;

    // Set the color -- be sure to set alpha to something non-zero!
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0;

    marker.lifetime = ros::Duration();

}

