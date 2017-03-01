#include <iostream>
#include <fstream>
#include <sstream>
#include <g2o/types/slam3d/types_slam3d.h> //node type
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <boost/shared_ptr.hpp>

#include "slam_base.h"

class CG2OBridge 
{
	// parameter solers. one depend on another
	typedef g2o::BlockSolver_6_3 SLAMBlockSolver;
	typedef g2o::LinearSolverCSparse<SLAMBlockSolver::PoseMatrixType> SLAMLinearSolver;

	SLAMLinearSolver* _linear_solver;
	SLAMBlockSolver* _block_solver;

	g2o::OptimizationAlgorithmLevenberg* _solver;

	//final optimizer to use!
	g2o::SparseOptimizer _global_optimizer;

	int _idx_start;
	int _idx_end;

	int _idx_cur;
	int _idx_lst;

	CG2OBridge() {
		_linear_solver = new SLAMLinearSolver();
		_linear_solver->setBlockOrdering(false);
		_block_solver = new SLAMBlockSolver(_linear_solver);
		_solver = new g2o::OptimizationAlgorithmLevenberg(_block_solver);
		_global_optimizer.setAlgorithm(_solver);

		_global_optimizer.setVerbose(false);
		_idx_start = 0;
		_idx_end = 0;
		_idx_cur = 0;
		_idx_lst = 0;
	}

	~CG2OBridge () {}

	// add the first node 
	void addFirstVertex() {
		_idx_lst = _idx_cur;
		g2o::VertexSE3* v = new g2o::VertexSE3();
		v->setId(_idx_cur);
		v->setEstimate(Eigen::Isometry3d::Identity()); 
		v->setFixed(true);
		_global_optimizer.addVertex(v);
	}

	void addAnVertex(const cv::Mat& rvec, const cv::Mat& tvec) {
		// add default vertex
		g2o::VertexSE3 *v = new g2o::VertexSE3();
		v->setId(_idx_cur); // set the current ID onlt
		Eigen::Isometry3d T = cvMat2Eigen(rvec, tvec);
		v->setEstimate(T); // add camera pose model
		_global_optimizer.addVertex(v);
		_idx_lst = _idx_cur;
		_idx_cur++;
	}

	// no vertex parameter because we do not care about the vertex, so we set default vertex to be I and
	// we focus on the edge which describe the motion. The parameter is the Edge
	void addNewVertexAndEdge(const cv::Mat& rvec, const cv::Mat& tvec, int idx0, int idx1) {
 		// add edge
		g2o::EdgeSE3 *edge = new g2o::EdgeSE3();
		edge->vertices()[0] = _global_optimizer.vertex(idx0);
		edge->vertices()[1] = _global_optimizer.vertex(idx1);

		//information cv::Matrix. the conv of two observations
		Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Identity();
		information(0,0) = information(1,1) = information(2,2) = 100;
		information(3,3) = information(4,4) = information(5,5) = 100;
		edge->setInformation(information);
		Eigen::Isometry3d T = cvMat2Eigen(rvec, tvec);
		edge->setMeasurement(T);
	}

	// convert the rotation and translation vectors from cv::Mat to Eigen::Isometry3d
	Eigen::Isometry3d cvMat2Eigen(const cv::Mat& rvec, const cv::Mat& tvec)
	{
	    cv::Mat R;
	    Rodrigues( rvec, R );
	    Eigen::Matrix3d r;
	    for (int i = 0; i < 3; i++) {
	    	for (int j = 0; j < 3; j++) {
	    		r(i,j) = R.at<double>(i,j);
	    	}
	    }
	  
	    Eigen::Isometry3d Tr = Eigen::Isometry3d::Identity();

	    Eigen::AngleAxisd angle(r);
	    Eigen::Translation<double,3> trans(tvec.at<double>(0,0), tvec.at<double>(0,1), tvec.at<double>(0,2));
	    Tr = angle;
	    Tr(0,3) = tvec.at<double>(0,0); 
	    Tr(1,3) = tvec.at<double>(0,1); 
	    Tr(2,3) = tvec.at<double>(0,2);
	    return Tr;
	}

	void OptimizeProcess(std::string dir_to_save = "/home/rokid/Documents/g2o/", int iter = 100)
	{
		std::cout << "start to G.O., vertice num:" << _global_optimizer.vertices().size() << endl;
		std::string before_save_file = dir_to_save + "result_before_go.g2o";
		std::string after_save_file = dir_to_save + "result_after_go.g20";
		_global_optimizer.save(before_save_file.c_str());
		_global_optimizer.initializeOptimization();
		_global_optimizer.optimize(iter);
		_global_optimizer.save(after_save_file.c_str());
	}
 };


using namespace std;
using namespace cv;