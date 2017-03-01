#ifndef TRACKER_H_
#define TRACKER_H_

// this file will define a class to compute the relative changes between two neighbo frames
// the tracker will be a static instance which works as a member of the 

#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include "rgbd_frame.h"

using namespace std;
using namespace cv;

class CRGBDFrame;
class CCameraPara;

class CTracker
{
public:
	CCameraPara _rgbcam;
	CCameraPara _depthcam;

	cv::Mat _Kdepth;
	cv::Mat _Krgb;
	cv::Mat _R;
	cv::Mat _T;

	std::vector<CRGBDFrame> _frames;
	CRGBDFrame _frame_fst;
	CRGBDFrame _frame_lst;
	CRGBDFrame _frame_cur;

public:
	CTracker()
	{
	
	}

	CTracker(CParameterReader para)
	{
  		// init rgb camera
  		_rgbcam = CCameraPara(	para.rgb_camera_width, para.rgb_camera_height,
  								para.rgb_camera_fx, para.rgb_camera_fy, para.rgb_camera_cx, para.rgb_camera_cy);
  		double kinect_rgb_camera_intrinsic_para[3][3] =  { { _rgbcam._fx,           0, _rgbcam._cx},
								  						   {           0, _rgbcam._fy, _rgbcam._cy},
								  						   {           0,	   	    0,           1} }; 
  		_Krgb = cv::Mat(3, 3, CV_64F);
  		for (int i = 0; i < _Krgb.rows; i++)
  			for (int j = 0; j < _Krgb.cols; j++)
  				_Krgb.at<double>(i, j) = kinect_rgb_camera_intrinsic_para[i][j];

		// init depth camera
		_depthcam = CCameraPara(para.depth_camera_width, para.depth_camera_height,
								para.depth_camera_fx, para.depth_camera_fy, para.depth_camera_cx, para.depth_camera_cy);
  		double kinect_depth_camera_intrinsic_para[3][3] =  { { _depthcam._fx,              0, _depthcam._cx},
								  							 {             0,  _depthcam._fy, _depthcam._cy},
								  							 {             0,		       0,	  	      1} }; 
  		_Kdepth = cv::Mat(3, 3, CV_64F);
  		for (int i = 0; i < _Kdepth.rows; i++)
  			for (int j = 0; j < _Kdepth.cols; j++)
  				_Kdepth.at<double>(i, j) = kinect_depth_camera_intrinsic_para[i][j];

  		// init R and T 
  		_R = cv::Mat::zeros(3, 1, CV_64F);
  		_T = cv::Mat::zeros(3, 1, CV_64F);

	}

	~CTracker()
	{
		_R.release();
		_T.release();
	}

	int setReferenceFrame(CRGBDFrame frame_reference) {
		namedWindow("line");
		moveWindow("line", 1600,900);
		_frame_fst = frame_reference;
		_frame_lst = _frame_fst;
	}

	int track (const CRGBDFrame& frame_lst, const CRGBDFrame& frame_cur)
	{
		//trackMono(frame_lst._img_rgb, frame_cur._img_rgb, _R, _T);
		int res = trackRGBPoint(frame_lst._img_rgb, 
							      frame_lst._keypoints, frame_lst._kps_3d, frame_lst._descriptor,
							      frame_cur._img_rgb,
			      		 		  _R, _T);
		return res;
	}

	/// this is a function that track the R,T movement frame by frame and renew the R,T gradually. 
	/// The last key frame and the key points, 3d points are updated frame by frame.
	int trackIter(const CRGBDFrame& frame_cur)
	{
		// Steps:
		// (1) import the current frame (only)
		// (2) track the last frame, get the key points
		// (3) update the last frame, renew the image, keypoints, kps_3d, descriptor
		// only renew the data before optimizing matching !!!
		int res = trackRGBPoint(_frame_fst._img_rgb, _frame_fst._keypoints, _frame_fst._kps_3d, _frame_fst._descriptor,
							      frame_cur._img_rgb,
			      		 		  _R, _T);

		return res;
	}

	/// track the camera using two neighbour color images (actually grayscale images)
	/// output rotation and translation matrix
	int trackMono(const cv::Mat& img_color_lst, const cv::Mat& img_color_cur, cv::Mat& R, cv::Mat& T)
	{
		// rgb convert to grayscale import the point cloud information
		// get all the orgin data done in this step
		cv::Mat img_gray_lst;
		cv::Mat img_gray_cur;
		cv::cvtColor(img_color_lst, img_gray_lst, CV_BGR2GRAY);
		cv::cvtColor(img_color_cur, img_gray_cur, CV_BGR2GRAY);

		cv::Mat img_lst = img_gray_lst;
		cv::Mat img_cur = img_gray_cur;

		// denoising, smooth the image, remove the p&s noise in the color image 
		// (TODO remove noise point in the point cloud)
		cv::medianBlur(img_lst, img_lst, 3);
		cv::medianBlur(img_cur, img_cur, 3);

		//define key points
		std::vector<cv::KeyPoint> kps_lst, kps_cur;
		cv::Mat dscp_lst, dscp_cur;

		// use orb feature detector to extract keypoints and descriptors. 
		// the parameters are default
		cv::ORB orb(1000, 1.2, 4, 31, 0, 2, ORB::HARRIS_SCORE, 31);
		orb.detect(img_lst, kps_lst);
		orb.detect(img_cur, kps_cur);
		orb.compute(img_lst, kps_lst, dscp_lst);
		orb.compute(img_cur, kps_cur, dscp_cur);

		//Display key points information (((O_O we do not need to show a single key points frame. we will 
		// later show the matches of the two images)))
		//#define SHOW1
		#ifdef SHOW1
		cv::Mat img_show1, img_show2;
	    cv::drawKeypoints( img_lst, kps_lst, img_show1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT ); // DRAW_RICH_KEYPOINTS
	    cv::imshow( "keypoints1", img_show1 );
	    cv::drawKeypoints( img_cur, kps_cur, img_show2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
	    cv::imshow( "keypoints2", img_show2 );
	    cv::waitKey(0); 
	    img_show1.release();
	    img_show2.release();
	    #endif
	    //Display done
	    
	    // initial match key points: find and matching pairs
	    std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(NORM_HAMMING, true);
	    matcher.match(dscp_cur, dscp_lst, matches);
	    if (matches.size() < 10)
			return -1;

	    // rank the matches(TODO) and only keep the good match
		double max_dist = 0; double min_dist = 256;
		//-- Quick calculation of max and min distances between keypoints
		for( int i = 0; i < matches.size(); i++ ){ 
			double dist = matches[i].distance;

			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
		}
		std::cout << "-- Max dist :" << max_dist << std::endl;
		std::cout << "-- Min dist :" << min_dist << std::endl;
		double threshold = 2*min_dist > 32? 2*min_dist:32;

		// extract good matches by rank distances of pixel and spatial
  		std::vector<cv::DMatch> good_matches;
  		int count = 0;
		for( int i = 0; i < matches.size(); i++ ){
			if( matches[i].distance <= threshold) {
			  	cv::Point2d id_lst, id_cur;
	  			id_lst = kps_lst[matches[i].trainIdx].pt;
	  			id_cur = kps_cur[matches[i].queryIdx].pt;

	  			if (cv::norm(id_lst - id_cur) < 100) { 
	  				good_matches.push_back(matches[i]);
	  				count++;
	  			}
		    } 
		}
		std::cout << "good_matches num: " << good_matches.size() << std::endl;
		if (good_matches.size() < 1)
			return -1;

		// Display final matches
		#define SHOW2
		#ifdef SHOW2
		cv::Mat img_match;
		cv::drawMatches(img_cur, kps_cur, img_lst, kps_lst, good_matches, img_match);
		cv::imshow("final matches", img_match);
		cv::waitKey(0);
	    #endif

		std::vector<cv::Point2d> ps_lst, ps_cur;
		//keyPointPairsToPointVectorPairs(kps_lst, kps_cur, good_matches, ps_lst, ps_cur);
	    //transformFromFundamental(_Krgb, ps_lst, ps_cur);
	    //transformFromHomography(_Krgb, ps_lst, ps_cur);

  		return good_matches.size();
	}

	/// track the camera from the new coming picture to the reference frame (given key points, 3d point positions and ORB descriptor)
	/// output rotation and translation matrix
	/// Attention! it looks that the "const cv::Mat& img_color_lst" parameter is useless
	int trackRGBPoint(const cv::Mat& img_color_lst,  
				  const std::vector<cv::KeyPoint>& kps_lst, const std::vector<cv::Point3d>& pt3d_lst, const cv::Mat& dscp_lst,  
		          const cv::Mat& img_color_cur,
				  cv::Mat& R, cv::Mat& T)
	{
		// 0. convert the train/reference/last image to grayscale and smooth it. Only for displayment.
		cv::Mat img_gray_lst;
		cv::cvtColor(img_color_lst, img_gray_lst, CV_BGR2GRAY);
		cv::Mat img_lst = img_gray_lst;
		cv::medianBlur(img_lst, img_lst, 3);

		// 1. convert the current/query image to grayscale, smooth it
		cv::Mat img_gray_cur;
		cv::cvtColor(img_color_cur, img_gray_cur, CV_BGR2GRAY);
		cv::Mat img_cur = img_gray_cur;
		cv::medianBlur(img_cur, img_cur, 3);

		// 2. detect key points on the current image, extract the ORB features and compute descriptor for the current image
		cv::ORB orb(1000, 1.2, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
		std::vector<cv::KeyPoint> kps_cur;
		cv::Mat dscp_cur;
		orb.detect(img_cur, kps_cur);
		orb.compute(img_cur, kps_cur, dscp_cur);

		// 3. match two frames, get the matching 
		std::vector<cv::DMatch> matches;
	    cv::BFMatcher matcher(NORM_HAMMING, true);
	    matcher.match(dscp_cur, dscp_lst, matches);
	    if (matches.size() < 10)
			return -1;

		// 4. filter low score match, only keep the high score matches
		//	  we will have several filters. 1st from score, 2nd from fake optical flow
		//    Here we will compute a fake optical flow and abort all the points which does not match the flow
  		std::vector<cv::DMatch> good_matches;
  		int threshold = 96;

  		// select good score matching
  		std::vector<cv::DMatch> before_matches = matches;
  		std::vector<cv::DMatch> after_matches;
		for( int i = 0; i < before_matches.size(); i++ ){
			if( before_matches[i].distance <= threshold) {
				after_matches.push_back(before_matches[i]);
			}
		}
		std::cout << "after_matches num: " << after_matches.size() << std::endl;

		good_matches = after_matches;
		std::cout << "good_matches num: " << good_matches.size() << std::endl;
		if (good_matches.size() < 10)
			return -1;	

		// 5. compute the R,T use solveRansac. Given the 3D points and their two projections, the problem becomes a stereo camera calibration problem
		//    Oh yeah. 
		//  5.1 get 3D point
		std::vector<cv::Point3f> pt3d;
		for (int i = 0; i < good_matches.size(); i++) {
			pt3d.push_back(pt3d_lst[good_matches[i].trainIdx]);
		}
		//  5.2 get two 2d image pixels
		std::vector<cv::Point2f> pn0;
		std::vector<cv::Point2f> pn1;
		std::vector<int> idpn0;
		std::vector<int> idpn1;
		keyPointPairsToPointVectorPairs(kps_lst, kps_cur, good_matches, pn0, pn1, idpn0, idpn1);
		//  5.3 calculate RT
		cv::Mat distortionCoefficients;
	    cv::Mat rvector;
		cv::Mat tvector;
		cv::Mat inliers;

		//  5.4 calculate r and t // TODO, ransac is needed here!
		cv::solvePnPRansac(pt3d, pn1,
		               _Krgb, distortionCoefficients,
		               rvector, tvector, 
		               true, 200, 2.0, pt3d.size() / 2, inliers, CV_EPNP);

		displayKeypointsMatching(img_color_cur, pn0, pn1, inliers);

		if (inliers.rows < 7)
			return -2;

		std::cout << float(inliers.rows) << ", " <<  float(pn0.size()) << ", " << norm(rvector) << std::endl;
		if (float(inliers.rows) / float(pn0.size()) < 0.10 || norm(rvector) > 1.0) 
			return -3;

		cv::Mat rmatrix;
		cv::Mat tmatrix = tvector;
		eulerAnglesToRotationMatrix(rvector.at<double>(0,0), rvector.at<double>(1,0), rvector.at<double>(2,0), rmatrix);
		std::cout << rmatrix << std::endl;

		// rectify the rotation angle and translate to match the camera coordinate
		rvector.copyTo(_R);
		tvector.copyTo(_T);

		std::cout << "_R:\n" << _R << std::endl;
		std::cout << "_T:\n" << _T << std::endl;
 
		// update the _frame_lst with the information of the current frame. The system will keep on tracking the camera using the last information
		// triangulate pixels
		std::vector<cv::Point2d> im0, im1;
		std::vector<cv::Point3d> pt_gt;
		for (int i = 0; i < inliers.rows; i++) {
			im0.push_back(pn0[inliers.at<int>(i,0)]);
			im1.push_back(pn1[inliers.at<int>(i,0)]);
			pt_gt.push_back(pt3d[inliers.at<int>(i,0)]);
		}

		// triangulate using customer function which uses official API
		std::vector<cv::Point3d> pointcloud;
  		triangulatePixels(im0, im1, 
  						  _Krgb, cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F), rmatrix, tvector, 
  		                  pointcloud);

  		// for (int i = 0; i < im0.size(); i ++ ) {
  		// 	cout << im0[i] << "\t" << im1[i] << "\t" << pt_gt[i] << "\t" << pointcloud[i] << endl;
  		// }

	    return 0;
	}

	/// this function will calculate the transformation between the two neighbour frames
	/// using fundamental matrix
	int transformFromFundamental(const cv::Mat& K, const std::vector<cv::Point2d>& ps0, const std::vector<cv::Point2d>& ps1)
	{
		std::cout << "_Krgb: " << std::endl << _Krgb << std::endl;

		// compute fundamental matrix
		cv::Mat inliers;
		cv::Mat F = findFundamentalMat(ps0, ps1, FM_RANSAC, 3, 0.99, inliers);
		std::cout << "F: " << F << std::endl;
		scoreFundamentalMatrix(F, ps0, ps1, inliers);
		// compute essential matrix
		cv::Mat E = K.t() * F * K;
		std::cout << "E: " << E << std::endl;
		// compute rotation and translation matrix
		cv::SVD svd(E,SVD::MODIFY_A);
		cv::Mat u = svd.u;
		cv::Mat vt = svd.vt;
		cv::Mat s = svd.w;
		cv::Matx33d W(0, -1, 0, 1, 0, 0, 0, 0, 1);
		cv::Mat_<double> R1 = u * cv::Mat(W) * vt;
		cv::Mat_<double> R2 = u * cv::Mat(W).t() * vt;
		cv::Mat_<double> T1 = u.col(2);
		cv::Mat_<double> T2 = -u.col(2);
		if (!CheckCoherentRotation (R1)) {
			std::cout<<"resulting rotation is not coherent\n";
			return 0;
		}

		std::cout << "R1: " << std::endl << R1 << std::endl;
		std::cout << "T1: " << std::endl << T1 << std::endl << std::endl;

		std::cout << "R1: " << std::endl << R1 << std::endl;
		std::cout << "T2: " << std::endl << T2 << std::endl << std::endl;

		std::cout << "R2: " << std::endl << R2 << std::endl;
		std::cout << "T1: " << std::endl << T1 << std::endl << std::endl;

		std::cout << "R2: " << std::endl << R2 << std::endl;
		std::cout << "T2: " << std::endl << T2 << std::endl << std::endl;

		return 0;
	}

	/// this function will calculate the transformation between the two neighbour frames when most points are on a planar
	/// using homography matrix
	int transformFromHomography(const cv::Mat& K, const std::vector<cv::Point2d>& ps0, const std::vector<cv::Point2d>& ps1)
	{
		std::cout << "_Krgb: " << std::endl << _Krgb << std::endl;
		// compute homography matrix
		cv::Mat inliers;
		cv::Mat H = cv::findHomography(ps0, ps1, CV_RANSAC, 3, inliers);
		std::cout << "H: " << H << std::endl;
		scoreHomographyMatrix(H, ps0, ps1, inliers);
		// compute essential matrix
		cv::Mat A = K.inv() * H;
		std::cout << "A: " << A << std::endl;
		// compute rotation and translation matrix

		cv::Mat r1 = A.col(0);
		r1 = r1 / norm(r1);
		cv::Mat r2 = A.col(1);
		r2 = r2 / norm(r2);
		cv::Mat r3 = r1.clone();
		r3 = r3.cross(r2);
		cv::Mat R(3,3,CV_32F);
		r1.copyTo(R.col(0));
		r2.copyTo(R.col(1));
		r3.copyTo(R.col(2));

		cv::Mat T = A.col(2);
		

		std::cout << "R: " << std::endl << R << std::endl;
		std::cout << "T: " << std::endl << T << std::endl;

		return 0;
	}

	// check if the rotation is coherent
	bool CheckCoherentRotation(const cv::Mat_<double>& R) 
	{
    	if (fabsf(determinant(R)) - 1.0 > 1e-07) {
        	cerr << "rotation matrix is `1" << std::endl;
        	return false;
    	}
    	return true;
    }

    /// score the fundamental matrix from using the pixel error (used now, we do not like fundamental matrix)
    double scoreFundamentalMatrix(cv::Mat F, std::vector<cv::Point2d> p0, std::vector<cv::Point2d> p1, cv::Mat inliers)
    {
    	cv::Mat x0(3, p0.size(), F.type());
    	cv::Mat x1(3, p1.size(), F.type());

    	for (int i = 0; i < x0.cols; i++) {
    		x0.at<double>(0, i) = p0[i].x;
    		x0.at<double>(1, i) = p0[i].y;
    		x0.at<double>(2, i) = 1;
    	}
       	for (int i = 0; i < x1.cols; i++) {
    		x1.at<double>(0, i) = p1[i].x;
    		x1.at<double>(1, i) = p1[i].y;
    		x1.at<double>(2, i) = 1;
    	}

    	cv::Mat x1t = x1.t();
    	cv::Mat e_mat = x1t * F * x0;
    	double res = 0;
    	cv::Mat err = cv::Mat::zeros(x1.cols, 1, CV_64F);
    	int valid_num = sum(inliers)[0];
    	for (int i = 0; i < err.rows; i++)  {
    		if ( inliers.data[i] ) {
    			res += fabs(e_mat.at<double>(i, i));
    			err.at<double>(i, 0) = e_mat.at<double>(i, i);
    		}
    	}
    	//std::cout << err << std::endl;

    	res /= valid_num;
    	std::cout << "Fundamental Model Error: " << res << std::endl;
    	return res;
    }

    /// score the homography matrix from using the pixel error (not used now, we prefer fundamental matrix)
    double scoreHomographyMatrix(cv::Mat H, std::vector<cv::Point2d> p0, std::vector<cv::Point2d> p1, cv::Mat inliers)
    {
    	cv::Mat x0(3, p0.size(), H.type());
    	cv::Mat x1(3, p1.size(), H.type());

    	for (int i = 0; i < x0.cols; i++) {
    		x0.at<double>(0, i) = p0[i].x;
    		x0.at<double>(1, i) = p0[i].y;
    		x0.at<double>(2, i) = 1;
    	}
       	for (int i = 0; i < x1.cols; i++) {
    		x1.at<double>(0, i) = p1[i].x;
    		x1.at<double>(1, i) = p1[i].y;
    		x1.at<double>(2, i) = 1;
    	}

    	cv::Mat x1_cal = H * x0;
    	//std::cout << x1.t() << std::endl;
    	//std::cout << x1_cal.t() << std::endl;

    	float valid_num = sum(inliers)[0];
    	cv::Mat err = cv::Mat::zeros(x1.cols, 1, CV_64F);
    	for (int i = 0; i < x1.cols; i++) {
    		cv::Point2d dp(x1_cal.at<double>(0,i) - x1.at<double>(0,i),
    				   x1_cal.at<double>(1,i) - x1.at<double>(1,i) );
			if ( inliers.data[i] )
    			err.at<double>(i, 0) = norm(dp);
    	}

    	//std::cout << err << std::endl;
		double res = sum(err)[0] / valid_num;
    	std::cout << "homography Model Error: " << res << std::endl;
    	return 0;
    }

    /// get the cv::Point2d type key points pair list from two key points set with their matchings
    /// then the two lists of points are in the same length
	void keyPointPairsToPointVectorPairs(std::vector<cv::KeyPoint> kpt, std::vector<cv::KeyPoint> kpq, std::vector<DMatch> m, 
		                                 std::vector<cv::Point2f>& pst, std::vector<cv::Point2f>& psq,
		                                 std::vector<int>& idxt, std::vector<int>& idxq)
	{
		for (int i = 0; i < m.size(); i++) {
			cv::Point2d pt, pq;
		  	pt = kpt[m[i].trainIdx].pt;
		  	pq = kpq[m[i].queryIdx].pt;

		  	pst.push_back(pt);
		  	idxt.push_back(m[i].trainIdx);

		  	psq.push_back(pq);
		  	idxq.push_back(m[i].queryIdx);
	  	}
	}

	/// triangulate points
  	void triangulatePixels(const std::vector<cv::Point2d>& im0, const std::vector<cv::Point2d>& im1, 
  							 const cv::Mat& K, const cv::Mat& R0, const cv::Mat& T0, const cv::Mat& R1, const cv::Mat& T1, 
  		                     std::vector<cv::Point3d>& pointcloud)
	{
  		// triangulate using official
		cv::Mat RT0(3,4,CV_64F);
		R0.copyTo(RT0(cv::Rect(0, 0, 3, 3)));
		T0.copyTo(RT0(cv::Rect(3, 0, 1, 3)));
		cv::Mat RT1(3,4,CV_64F);
		R1.copyTo(RT1(cv::Rect(0, 0, 3, 3)));
		T1.copyTo(RT1(cv::Rect(3, 0, 1, 3)));
		cv::Mat pc4dmat(4,im0.size(),CV_64FC4);
  		triangulatePoints(_Krgb*RT0, _Krgb*RT1, im0, im1, pc4dmat);
		pointcloud.resize(im0.size());

  		for (int i = 0; i < pointcloud.size(); i++) {
  			pointcloud[i].x = pc4dmat.at<double>(0,i) / pc4dmat.at<double>(3,i);
  			pointcloud[i].y = pc4dmat.at<double>(1,i) / pc4dmat.at<double>(3,i);
  			pointcloud[i].z = pc4dmat.at<double>(2,i) / pc4dmat.at<double>(3,i);
  		}

  		RT0.release();
  		RT1.release();
	}

	/// change axis. opt is a string with 3 letter, xyz or zxy, or.....
	void changeAxis(std::vector<cv::Point3d>& pts, string opt)
	{
		std::vector<cv::Point3d> res = pts;

		// X axis
		if (opt[0] == 'x') 
			for (int i = 0; i < pts.size(); i++)
				res[i].x = pts[i].x;
		if (opt[0] == 'y') 
			for (int i = 0; i < pts.size(); i++)
				res[i].x = pts[i].y;
		if (opt[0] == 'z') 
			for (int i = 0; i < pts.size(); i++)
				res[i].x = pts[i].z;

		// Y axis
		if (opt[1] == 'x') 
			for (int i = 0; i < pts.size(); i++)
				res[i].y = pts[i].x;
		if (opt[1] == 'y') 
			for (int i = 0; i < pts.size(); i++)
				res[i].y = pts[i].y;
		if (opt[1] == 'z') 
			for (int i = 0; i < pts.size(); i++)
				res[i].y = pts[i].z;

		// Z axis
		if (opt[2] == 'x') 
			for (int i = 0; i < pts.size(); i++)
				res[i].z = pts[i].x;
		if (opt[2] == 'y') 
			for (int i = 0; i < pts.size(); i++)
				res[i].z = pts[i].y;
		if (opt[2] == 'z') 
			for (int i = 0; i < pts.size(); i++)
				res[i].z = pts[i].z;

		pts = res;
	}

	/// coordinate to camera
	std::vector<cv::Point3d> changeAxisToCamera(const std::vector<cv::Point3d>& pts)
	{
		std::vector<cv::Point3d> res = pts;

		// for (int i = 0; i < pts.size(); i++)
		// 	res[i].x = pts[i].x;
		for (int i = 0; i < pts.size(); i++)
			res[i].y = -pts[i].y;
		// for (int i = 0; i < pts.size(); i++)
		// 	res[i].z = pts[i].z;

		return res;
	}

	void displayKeypointsMatching(cv::Mat img, std::vector<cv::KeyPoint> kps_lst, std::vector<cv::KeyPoint> kps_cur, std::vector<DMatch> matches)
	{
		// Display th corresponding points
		cv::Mat image_show = img.clone();
		for (int i = 0; i < matches.size(); i++) {
			cv::circle(image_show, kps_lst[matches[i].trainIdx].pt, 2, cv::Scalar(255, 0, 0), 1, CV_AA, 0);
			cv::circle(image_show, kps_cur[matches[i].queryIdx].pt, 2, cv::Scalar(0, 0, 255), 1, CV_AA, 0);
			cv::line(image_show, kps_cur[matches[i].queryIdx].pt, kps_lst[matches[i].trainIdx].pt, cv::Scalar(0, 255, 0), 1, CV_AA, 0);
		}
		cv::imshow("line", image_show);
		cv::waitKey(1);
		image_show.release();
	}

	void displayKeypointsMatching(cv::Mat img, std::vector<cv::Point2f> pn_lst, std::vector<cv::Point2f> pn_cur, cv::Mat inliers)
	{
		// Display th corresponding points
		cv::Mat image_show = img.clone();
		for (int i = 0; i < inliers.rows; i++) {
			int id = inliers.at<int>(i, 0);
			cv::circle(image_show, pn_lst[id], 2, cv::Scalar(255, 0, 0), 1, CV_AA, 0);
			cv::circle(image_show, pn_cur[id], 2, cv::Scalar(0, 0, 255), 1, CV_AA, 0);
			cv::line(image_show, pn_lst[id], pn_cur[id], cv::Scalar(0, 255, 0), 1, CV_AA, 0);
		}
		cv::imshow("line", image_show);
		cv::waitKey(1);
		image_show.release();
	}

	void renewTheLastRGBDFrame(cv::Mat img_rgb_cur, 
		                       std::vector<cv::KeyPoint> keypoints_cur, cv::Mat descriptor_cur, 
		                       std::vector<cv::Point3d> pt3d_lst, 
		                       std::vector<int> idx_lst, std::vector<int> idx_cur,
		                       cv::Mat inlier) {
		
		// define data to renew 
		cv::Mat img_rgb_new = img_rgb_cur.clone();
		std::vector<cv::KeyPoint> kps_new;
		std::vector<cv::Point3d> pt3d_new;
		cv::Mat dscp_new = cv::Mat::zeros(inlier.rows, 32, CV_8U);

		for (int i = 0; i < inlier.rows; i++) {
			int k = inlier.at<int>(i, 0);
			kps_new.push_back(  keypoints_cur[idx_cur[k]] );
			pt3d_new.push_back(      pt3d_lst[idx_lst[k]] );
			descriptor_cur.row(idx_cur[k]).copyTo(dscp_new.row(i));
		}

		_frame_lst.updateFrameFeatureData(img_rgb_new, kps_new, pt3d_new, dscp_new);
	}

	cv::Mat eulerAnglesToRotationMatrix(double x, double y, double z, cv::Mat& R)
	{
	    // Calculate rotation about x axis
	    cv::Mat R_x = (Mat_<double>(3,3) <<
	               1,       0,              0,
	               0,       cos(x),   -sin(x),
	               0,       sin(x),   cos(x)
	               );
	     
	    // Calculate rotation about y axis
	    cv::Mat R_y = (Mat_<double>(3,3) <<
	               cos(y),    0,      sin(y),
	               0,               1,      0,
	               -sin(y),   0,      cos(y)
	               );
	     
	    // Calculate rotation about z axis
	    cv::Mat R_z = (Mat_<double>(3,3) <<
	               cos(z),    -sin(z),      0,
	               sin(z),    cos(z),       0,
	               0,               0,                  1);
	     
	     
	    // Combined rotation matrix
	    cv::Mat Rm = R_z * R_y * R_x;
	    Rm.copyTo(R);
	    return R;
	 
	}
};



#endif